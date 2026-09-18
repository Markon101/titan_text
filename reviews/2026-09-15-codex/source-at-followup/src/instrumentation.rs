use anyhow::Result;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

/// Detailed metrics collected at a single latent recurrence tick or token step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TickMetrics {
    /// Global latent tick index
    pub tick: usize,
    /// Token sequence step index (if in sequential token mode)
    #[serde(default)]
    pub token_step: Option<usize>,
    /// L2 norm of the full continuous state: ||x||_2
    pub state_norm: f32,
    /// Displacement update magnitude: ||x_{t+1} - x_t||_2
    pub update_magnitude: f32,
    /// Cosine similarity to the state at the previous tick: cos(x_t, x_{t-1})
    pub cosine_to_prev: f32,
    /// Cosine similarity to the initial state: cos(x_t, x_0)
    pub cosine_to_origin: f32,
    /// Mean gate activation value in [0, 1]
    pub gate_mean: f32,
    /// Standard deviation of gate activations
    pub gate_std: f32,
    /// Fraction of gates saturated closed (< 0.10)
    pub gate_sat_closed: f32,
    /// Fraction of gates saturated open (> 0.90)
    pub gate_sat_open: f32,
    /// Fraction of update deltas saturated in tanh (|delta| > 0.95)
    pub delta_sat: f32,
    /// Effective dimensionality / participation ratio: (sum sigma_c^2)^2 / sum sigma_c^4
    pub effective_dimension: f32,
    /// MLP intermediate feature norm ||h1||_2
    pub mlp_norm: f32,
    /// Raw directional update norm ||delta||_2
    pub raw_delta_norm: f32,
    /// Gated update norm ||gate * delta||_2
    pub gated_delta_norm: f32,
    /// Physical viscous dissipation damping norm (if viscosity > 0)
    pub dissipation_norm: f32,
    /// Readout prediction: top token index
    pub top_token_id: usize,
    /// Readout prediction: decoded character or token representation
    pub top_token_char: String,
    /// Readout confidence: maximum softmax probability max_v p(v)
    pub confidence: f32,
    /// Readout prediction entropy: H = - sum p * ln(p)
    pub output_entropy: f32,
    /// KL divergence of readout distribution from previous tick: D_KL(P_t || P_{t-1})
    pub readout_kl_prev: f32,
}

/// Trace collecting metrics across ticks and providing export methods.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InstrumentationTrace {
    pub label: String,
    pub ticks: Vec<TickMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceSummary {
    pub total_ticks: usize,
    pub initial_norm: f32,
    pub final_norm: f32,
    pub mean_step_update: f32,
    pub max_step_update: f32,
    pub total_trajectory_length: f32,
    pub initial_confidence: f32,
    pub final_confidence: f32,
    pub initial_entropy: f32,
    pub final_entropy: f32,
    pub dead_gates_detected: bool,
    pub saturated_gates_detected: bool,
}

impl InstrumentationTrace {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            ticks: Vec::new(),
        }
    }

    pub fn record(&mut self, metric: TickMetrics) {
        self.ticks.push(metric);
    }

    pub fn summary(&self) -> TraceSummary {
        let total_ticks = self.ticks.len();
        if total_ticks == 0 {
            return TraceSummary {
                total_ticks: 0,
                initial_norm: 0.0,
                final_norm: 0.0,
                mean_step_update: 0.0,
                max_step_update: 0.0,
                total_trajectory_length: 0.0,
                initial_confidence: 0.0,
                final_confidence: 0.0,
                initial_entropy: 0.0,
                final_entropy: 0.0,
                dead_gates_detected: false,
                saturated_gates_detected: false,
            };
        }

        let initial_norm = self.ticks[0].state_norm;
        let final_norm = self.ticks.last().unwrap().state_norm;
        let initial_confidence = self.ticks[0].confidence;
        let final_confidence = self.ticks.last().unwrap().confidence;
        let initial_entropy = self.ticks[0].output_entropy;
        let final_entropy = self.ticks.last().unwrap().output_entropy;

        let mut sum_update = 0.0f32;
        let mut max_update = 0.0f32;
        let mut dead_gates = false;
        let mut sat_gates = false;

        for t in &self.ticks {
            sum_update += t.update_magnitude;
            if t.update_magnitude > max_update {
                max_update = t.update_magnitude;
            }
            if t.gate_sat_closed > 0.95 {
                dead_gates = true;
            }
            if t.gate_sat_open > 0.95 {
                sat_gates = true;
            }
        }

        TraceSummary {
            total_ticks,
            initial_norm,
            final_norm,
            mean_step_update: sum_update / total_ticks.max(1) as f32,
            max_step_update: max_update,
            total_trajectory_length: sum_update,
            initial_confidence,
            final_confidence,
            initial_entropy,
            final_entropy,
            dead_gates_detected: dead_gates,
            saturated_gates_detected: sat_gates,
        }
    }

    /// Serializes trace to a compact JSON string.
    #[allow(dead_code)]
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Serializes trace to JSONL (one JSON object per tick line).
    #[allow(dead_code)]
    pub fn to_jsonl(&self) -> Result<String> {
        let mut lines = Vec::with_capacity(self.ticks.len());
        for tick in &self.ticks {
            lines.push(serde_json::to_string(tick)?);
        }
        Ok(lines.join("\n"))
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatedTraceReport {
    pub total_traces: usize,
    pub mean_initial_norm: f32,
    pub mean_final_norm: f32,
    pub mean_step_update: f32,
    pub mean_trajectory_length: f32,
    pub mean_final_confidence: f32,
    pub mean_final_entropy: f32,
}

/// Helper function to compute participation ratio / effective dimensionality from a [B, L, C] tensor.
/// Optimized for mobile CPU cache locality using a contiguous single-allocation buffer.
pub fn compute_effective_dimension(tensor: &Tensor) -> Result<f32> {
    let (b, l, c) = tensor.dims3()?;
    if c == 0 {
        return Ok(0.0);
    }
    let n_rows = b * l;
    let n = n_rows as f32;
    if n <= 1.0 {
        return Ok(1.0);
    }
    let flat = tensor.flatten_all()?.to_vec1::<f32>()?;

    // 1. Channel means
    let mut channel_means = vec![0.0f32; c];
    for chunk in flat.chunks_exact(c) {
        for (ch, &val) in chunk.iter().enumerate() {
            channel_means[ch] += val;
        }
    }
    for mean in channel_means.iter_mut() {
        *mean /= n;
    }

    // 2. Center data
    let mut centered = flat;
    for chunk in centered.chunks_exact_mut(c) {
        for (ch, val) in chunk.iter_mut().enumerate() {
            *val -= channel_means[ch];
        }
    }

    // 3. Full C x C covariance matrix: Sigma = (X^T * X) / (N - 1)
    let mut cov = vec![0.0f32; c * c];
    let norm_factor = 1.0 / (n - 1.0);

    for chunk in centered.chunks_exact(c) {
        for i in 0..c {
            let xi = chunk[i];
            let row_offset = i * c;
            for j in 0..c {
                cov[row_offset + j] += xi * chunk[j];
            }
        }
    }
    for val in cov.iter_mut() {
        *val *= norm_factor;
    }

    // 4. Trace(Sigma) = sum of diagonal elements
    let mut trace_cov = 0.0f32;
    for i in 0..c {
        trace_cov += cov[i * c + i];
    }

    // 5. Frobenius norm squared: ||Sigma||_F^2 = sum_ij Sigma_ij^2 = Tr(Sigma^2)
    let mut frobenius_sq = 0.0f32;
    for &val in cov.iter() {
        frobenius_sq += val * val;
    }

    if frobenius_sq > 1e-12 {
        Ok((trace_cov * trace_cov) / frobenius_sq)
    } else {
        Ok(1.0)
    }
}

/// Aggregates multiple instrumentation traces across parallel trials/seeds into a consolidated report.
#[allow(dead_code)]
pub fn aggregate_summaries(traces: &[InstrumentationTrace]) -> AggregatedTraceReport {
    use rayon::prelude::*;
    let n = traces.len().max(1) as f32;
    let summaries: Vec<TraceSummary> = traces.par_iter().map(|t| t.summary()).collect();
    let mean_init_norm = summaries.par_iter().map(|s| s.initial_norm).sum::<f32>() / n;
    let mean_final_norm = summaries.par_iter().map(|s| s.final_norm).sum::<f32>() / n;
    let mean_update = summaries.par_iter().map(|s| s.mean_step_update).sum::<f32>() / n;
    let mean_len = summaries.par_iter().map(|s| s.total_trajectory_length).sum::<f32>() / n;
    let mean_conf = summaries.par_iter().map(|s| s.final_confidence).sum::<f32>() / n;
    let mean_ent = summaries.par_iter().map(|s| s.final_entropy).sum::<f32>() / n;

    AggregatedTraceReport {
        total_traces: traces.len(),
        mean_initial_norm: mean_init_norm,
        mean_final_norm: mean_final_norm,
        mean_step_update: mean_update,
        mean_trajectory_length: mean_len,
        mean_final_confidence: mean_conf,
        mean_final_entropy: mean_ent,
    }
}

/// Helper function to compute cosine similarity between two tensors of matching shape.
pub fn compute_cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    let norm_a = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_b = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let denom = norm_a * norm_b;
    if denom > 1e-8 {
        Ok((dot / denom).clamp(-1.0, 1.0))
    } else {
        Ok(1.0)
    }
}

/// Helper function to compute KL divergence between two probability distributions: D_KL(P || Q)
pub fn compute_kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    let mut kl = 0.0f32;
    for (&p_val, &q_val) in p.iter().zip(q.iter()) {
        if p_val > 1e-8 && q_val > 1e-8 {
            kl += p_val * (p_val / q_val).ln();
        }
    }
    kl.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_cosine_similarity() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::new(&[1.0f32, 0.0, 0.0], &dev)?;
        let b = Tensor::new(&[1.0f32, 0.0, 0.0], &dev)?;
        let c = Tensor::new(&[0.0f32, 1.0, 0.0], &dev)?;
        let d = Tensor::new(&[-1.0f32, 0.0, 0.0], &dev)?;

        assert!((compute_cosine_similarity(&a, &b)? - 1.0).abs() < 1e-5);
        assert!((compute_cosine_similarity(&a, &c)? - 0.0).abs() < 1e-5);
        assert!((compute_cosine_similarity(&a, &d)? - (-1.0)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_effective_dimension() -> Result<()> {
        let dev = Device::Cpu;
        // 4 orthogonal, uncorrelated channels -> effective dimension should be 4.0
        let data = vec![
            vec![vec![ 1.0f32,  0.0,  0.0,  0.0]],
            vec![vec![-1.0f32,  0.0,  0.0,  0.0]],
            vec![vec![ 0.0f32,  1.0,  0.0,  0.0]],
            vec![vec![ 0.0f32, -1.0,  0.0,  0.0]],
            vec![vec![ 0.0f32,  0.0,  1.0,  0.0]],
            vec![vec![ 0.0f32,  0.0, -1.0,  0.0]],
            vec![vec![ 0.0f32,  0.0,  0.0,  1.0]],
            vec![vec![ 0.0f32,  0.0,  0.0, -1.0]],
        ];
        let t = Tensor::new(data, &dev)?;
        let ed = compute_effective_dimension(&t)?;
        assert!((ed - 4.0).abs() < 0.2, "Expected ~4.0, got {ed}");
        Ok(())
    }

    #[test]
    fn test_instrumentation_trace_summary() -> Result<()> {
        let mut trace = InstrumentationTrace::new("test_trace");
        trace.record(TickMetrics {
            tick: 0,
            token_step: None,
            state_norm: 1.0,
            update_magnitude: 0.1,
            cosine_to_prev: 1.0,
            cosine_to_origin: 1.0,
            gate_mean: 0.5,
            gate_std: 0.1,
            gate_sat_closed: 0.0,
            gate_sat_open: 0.0,
            delta_sat: 0.0,
            effective_dimension: 10.0,
            mlp_norm: 1.0,
            raw_delta_norm: 0.2,
            gated_delta_norm: 0.1,
            dissipation_norm: 0.0,
            top_token_id: 1,
            top_token_char: "a".to_string(),
            confidence: 0.8,
            output_entropy: 0.5,
            readout_kl_prev: 0.0,
        });

        trace.record(TickMetrics {
            tick: 1,
            token_step: None,
            state_norm: 1.1,
            update_magnitude: 0.05,
            cosine_to_prev: 0.99,
            cosine_to_origin: 0.99,
            gate_mean: 0.5,
            gate_std: 0.1,
            gate_sat_closed: 0.0,
            gate_sat_open: 0.0,
            delta_sat: 0.0,
            effective_dimension: 10.0,
            mlp_norm: 1.0,
            raw_delta_norm: 0.1,
            gated_delta_norm: 0.05,
            dissipation_norm: 0.0,
            top_token_id: 1,
            top_token_char: "a".to_string(),
            confidence: 0.85,
            output_entropy: 0.4,
            readout_kl_prev: 0.02,
        });

        let s = trace.summary();
        assert_eq!(s.total_ticks, 2);
        assert!((s.initial_norm - 1.0).abs() < 1e-5);
        assert!((s.final_norm - 1.1).abs() < 1e-5);
        assert!((s.mean_step_update - 0.075).abs() < 1e-5);
        assert!(!s.dead_gates_detected);

        let json = trace.to_json()?;
        assert!(json.contains("test_trace"));
        let jsonl = trace.to_jsonl()?;
        assert_eq!(jsonl.lines().count(), 2);
        Ok(())
    }

    #[test]
    fn test_effective_dimension_rank1_and_independent() -> Result<()> {
        let dev = Device::Cpu;
        let c = 8;
        let n = 20;

        // 1. Collinear rank-1 tensor: all channels are identical
        let mut col_data = Vec::with_capacity(n * c);
        for i in 0..n {
            let val = (i as f32) * 0.5;
            for _ in 0..c {
                col_data.push(val);
            }
        }
        let col_tensor = Tensor::from_vec(col_data, (1, n, c), &dev)?;
        let pr_col = compute_effective_dimension(&col_tensor)?;
        // Collinear rank-1 should yield PR = 1.0
        assert!((pr_col - 1.0).abs() < 1e-3, "Expected PR ~ 1.0 for rank-1 collinear channels, got {pr_col}");

        // 2. Orthogonal standard basis tensor: PR should be c
        let mut eye_data = vec![0.0f32; c * 2 * c];
        for i in 0..c {
            eye_data[i * c + i] = 1.0;
            eye_data[(c + i) * c + i] = -1.0;
        }
        let eye_tensor = Tensor::from_vec(eye_data, (1, 2 * c, c), &dev)?;
        let pr_eye = compute_effective_dimension(&eye_tensor)?;
        assert!((pr_eye - (c as f32)).abs() < 1e-2, "Expected PR ~ {c} for orthogonal channels, got {pr_eye}");

        Ok(())
    }
}
