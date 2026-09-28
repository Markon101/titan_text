//! Opt-in substrate laboratory. Legacy weights and field semantics are unchanged.
//!
//! Transport v1 uses synchronous old-state exchange: one tick crosses at most
//! one hierarchy edge OR one same-scale edge. Ordered child restriction retains
//! child parity before the learned bottleneck; it is not an invertible encoding.
use crate::config::{FieldConfig, NcaConfig};
use crate::field::MorphogenicField;
use crate::nca::NeuralCellularAutomaton;
use anyhow::{bail, ensure, Result};
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubstrateKind {
    LegacyNca,
    Transport,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubstrateConfig {
    pub kind: SubstrateKind,
    pub version: u32,
    /// Maximum scale count, including the fine scale; stop at length one.
    pub scales: usize,
    pub channels: usize,
    pub hidden_dim: usize,
    pub transport: bool,
    pub exchange_mode: String,
    pub boundary: String,
    pub state_mode: String,
    pub reaction_step: f64,
    pub exchange_step: f64,
    /// Elementwise hard bound, never a reduction over space or batch.
    pub state_bound: f64,
}

impl Default for SubstrateConfig {
    fn default() -> Self {
        Self {
            kind: SubstrateKind::Transport,
            version: 1,
            scales: 4,
            channels: 32,
            hidden_dim: 48,
            transport: true,
            exchange_mode: "learned_gated_ordered_pair".into(),
            boundary: "zero".into(),
            state_mode: "continuous".into(),
            reaction_step: 0.2,
            exchange_step: 0.2,
            state_bound: 4.0,
        }
    }
}

impl SubstrateConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.version == 1, "unsupported substrate version");
        ensure!(self.scales > 0, "scales must be positive");
        ensure!(self.channels > 0, "channels must be positive");
        ensure!(self.hidden_dim > 0, "hidden_dim must be positive");
        if self.kind == SubstrateKind::Transport {
            ensure!(
                self.channels.is_multiple_of(4),
                "transport channels must divide into equal M,L,R,X quarters"
            );
            ensure!(
                self.boundary == "zero",
                "transport v1 supports only zero boundaries"
            );
            ensure!(
                self.exchange_mode == "learned_gated_ordered_pair",
                "unsupported exchange mode"
            );
            ensure!(
                self.state_mode == "continuous",
                "transport v1 is continuous only"
            );
        }
        ensure!(
            self.reaction_step.is_finite() && (0.0..=1.0).contains(&self.reaction_step),
            "reaction_step must be finite in [0,1]"
        );
        ensure!(
            self.exchange_step.is_finite() && (0.0..=1.0).contains(&self.exchange_step),
            "exchange_step must be finite in [0,1]"
        );
        ensure!(
            self.state_bound.is_finite() && self.state_bound > 0.0,
            "state_bound must be finite and positive"
        );
        Ok(())
    }

    /// Allocation order: memory, left-moving, right-moving, reaction.
    pub fn channel_allocation(&self) -> [usize; 4] {
        [self.channels / 4; 4]
    }

    pub fn scale_lengths(&self, length: usize) -> Result<Vec<usize>> {
        self.validate()?;
        ensure!(length > 0, "empty sequences are unsupported");
        let mut lengths = vec![length];
        if self.kind == SubstrateKind::Transport {
            while lengths.len() < self.scales && *lengths.last().unwrap() > 1 {
                lengths.push(lengths.last().unwrap().div_ceil(2));
            }
        }
        Ok(lengths)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lesions {
    pub disable_transport: bool,
    pub disable_reaction: bool,
    pub disable_up: bool,
    pub disable_down: bool,
    /// Applied before the tick (repeated resets suppress outgoing state).
    pub reset_scale: Option<usize>,
    /// Cyclic batch derangement before the tick; requires batch >= 2.
    pub shuffle_scale: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum State {
    Legacy(MorphogenicField),
    Transport(Vec<Tensor>),
}

pub enum Substrate {
    Legacy(NeuralCellularAutomaton),
    Transport(TransportField),
}

pub struct TransportField {
    pub config: SubstrateConfig,
    reaction_hidden: Linear,
    reaction_delta: Linear,
    reaction_gate: Linear,
    up_delta: Linear,
    up_gate: Linear,
    down_delta: Linear,
    down_gate: Linear,
}

impl Substrate {
    pub fn new(
        vb: VarBuilder<'_>,
        config: &SubstrateConfig,
        nca: &NcaConfig,
        field: &FieldConfig,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.channels == field.channels,
            "substrate/field channel mismatch"
        );
        Ok(match config.kind {
            SubstrateKind::LegacyNca => {
                Self::Legacy(NeuralCellularAutomaton::new(vb.pp("nca"), nca, field)?)
            }
            SubstrateKind::Transport => {
                ensure!(
                    !field.periodic_boundary,
                    "transport requires explicit nonperiodic field config"
                );
                let vb = vb.pp("transport");
                let c = config.channels;
                Self::Transport(TransportField {
                    config: config.clone(),
                    reaction_hidden: linear(c, config.hidden_dim, vb.pp("reaction_hidden"))?,
                    reaction_delta: linear(config.hidden_dim, c, vb.pp("reaction_delta"))?,
                    reaction_gate: linear(config.hidden_dim, c, vb.pp("reaction_gate"))?,
                    up_delta: linear(2 * c, c, vb.pp("up_delta"))?,
                    up_gate: linear(2 * c, c, vb.pp("up_gate"))?,
                    down_delta: linear(c, 2 * c, vb.pp("down_delta"))?,
                    down_gate: linear(c, 2 * c, vb.pp("down_gate"))?,
                })
            }
        })
    }

    pub fn initialize(&self, input: Tensor) -> Result<State> {
        let (b, l, c) = input.dims3()?;
        ensure!(b > 0 && l > 0, "batch and length must be positive");
        match self {
            Self::Legacy(nca) => {
                ensure!(c == nca.field_cfg.channels, "input channel mismatch");
                let mut field = nca.field_cfg.clone();
                field.seq_len = l;
                Ok(State::Legacy(MorphogenicField::from_tensor(input, &field)))
            }
            Self::Transport(model) => {
                ensure!(c == model.config.channels, "input channel mismatch");
                let mut states = vec![input.clone()];
                for length in model.config.scale_lengths(l)?.into_iter().skip(1) {
                    states.push(Tensor::zeros(
                        (b, length, c),
                        input.dtype(),
                        input.device(),
                    )?);
                }
                Ok(State::Transport(states))
            }
        }
    }

    pub fn fine<'a>(&self, state: &'a State) -> &'a Tensor {
        match state {
            State::Legacy(s) => &s.x,
            State::Transport(s) => &s[0],
        }
    }

    pub fn states<'a>(&self, state: &'a State) -> Vec<&'a Tensor> {
        match state {
            State::Legacy(s) => std::iter::once(&s.x).chain(s.slow_state.iter()).collect(),
            State::Transport(s) => s.iter().collect(),
        }
    }

    pub fn step(&self, state: &State, lesions: &Lesions) -> Result<State> {
        match (self, state) {
            (Self::Legacy(nca), State::Legacy(s)) => {
                ensure!(
                    !lesions.disable_transport
                        && !lesions.disable_reaction
                        && !lesions.disable_up
                        && !lesions.disable_down,
                    "transport-specific lesions are not defined for legacy NCA"
                );
                let mut s = s.clone();
                if let Some(scale) = lesions.reset_scale {
                    ensure!(scale == 0, "legacy adapter reset supports only fine state");
                    s.x = s.x.zeros_like()?;
                }
                if let Some(scale) = lesions.shuffle_scale {
                    ensure!(
                        scale == 0,
                        "legacy adapter shuffle supports only fine state"
                    );
                    s.x = derange_batch(&s.x)?;
                }
                Ok(State::Legacy(nca.step_field(&s, s.x.device())?))
            }
            (Self::Transport(model), State::Transport(s)) => {
                Ok(State::Transport(model.step(s, lesions)?))
            }
            _ => bail!("state belongs to another substrate"),
        }
    }

    pub fn evolve(&self, state: &State, ticks: usize, lesions: &Lesions) -> Result<State> {
        let mut state = state.clone();
        for _ in 0..ticks {
            state = self.step(&state, lesions)?;
        }
        Ok(state)
    }
}

fn derange_batch(x: &Tensor) -> Result<Tensor> {
    let b = x.dim(0)?;
    ensure!(b >= 2, "batch shuffle requires at least two samples");
    Ok(Tensor::cat(
        &[x.narrow(0, 1, b - 1)?, x.narrow(0, 0, 1)?],
        0,
    )?)
}

/// Exact one-cell characteristic shift with absorbing (zero) boundaries.
fn shift(x: &Tensor, right: bool) -> Result<Tensor> {
    let (b, l, c) = x.dims3()?;
    let zero = Tensor::zeros((b, 1, c), x.dtype(), x.device())?;
    if l == 1 {
        return Ok(zero);
    }
    Ok(if right {
        Tensor::cat(&[zero, x.narrow(1, 0, l - 1)?], 1)?
    } else {
        Tensor::cat(&[x.narrow(1, 1, l - 1)?, zero], 1)?
    })
}

fn transport(x: &Tensor) -> Result<Tensor> {
    let q = x.dim(2)? / 4;
    Ok(Tensor::cat(
        &[
            x.narrow(2, 0, q)?,
            shift(&x.narrow(2, q, q)?, false)?,
            shift(&x.narrow(2, 2 * q, q)?, true)?,
            x.narrow(2, 3 * q, q)?,
        ],
        2,
    )?)
}

/// [child0 C, child1 C]; an absent odd final child is explicitly zero.
fn ordered_pairs(x: &Tensor) -> Result<Tensor> {
    let (b, l, c) = x.dims3()?;
    let padded = if l % 2 == 1 {
        Tensor::cat(
            &[x.clone(), Tensor::zeros((b, 1, c), x.dtype(), x.device())?],
            1,
        )?
    } else {
        x.clone()
    };
    Ok(padded.contiguous()?.reshape((b, l.div_ceil(2), 2 * c))?)
}

impl TransportField {
    fn step(&self, states: &[Tensor], lesions: &Lesions) -> Result<Vec<Tensor>> {
        ensure!(!states.is_empty(), "empty transport state");
        let (b, l, c) = states[0].dims3()?;
        let lengths = self.config.scale_lengths(l)?;
        ensure!(
            states.len() == lengths.len() && c == self.config.channels && b > 0,
            "state topology mismatch"
        );
        for (state, length) in states.iter().zip(&lengths) {
            ensure!(state.dims3()? == (b, *length, c), "scale shape mismatch");
        }
        let mut old = states.to_vec();
        if let Some(s) = lesions.reset_scale {
            ensure!(s < old.len(), "reset scale outside topology");
            old[s] = old[s].zeros_like()?;
        }
        if let Some(s) = lesions.shuffle_scale {
            ensure!(s < old.len(), "shuffle scale outside topology");
            old[s] = derange_batch(&old[s])?;
        }
        let mut result = Vec::with_capacity(old.len());
        for (s, x) in old.iter().enumerate() {
            let mut next = if self.config.transport && !lesions.disable_transport {
                transport(x)?
            } else {
                x.clone()
            };
            if !lesions.disable_reaction && self.config.reaction_step > 0.0 {
                let h = self.reaction_hidden.forward(&next.contiguous()?)?.tanh()?;
                let delta = self.reaction_delta.forward(&h)?.tanh()?;
                let gate = candle_nn::ops::sigmoid(&self.reaction_gate.forward(&h)?)?;
                next = (&next + ((delta * gate)? * self.config.reaction_step)?)?;
            }
            if self.config.exchange_step > 0.0 {
                if s > 0 && !lesions.disable_up {
                    let pair = ordered_pairs(&old[s - 1])?;
                    let delta = self.up_delta.forward(&pair)?.tanh()?;
                    let gate = candle_nn::ops::sigmoid(&self.up_gate.forward(&pair)?)?;
                    next = (&next + ((delta * gate)? * self.config.exchange_step)?)?;
                }
                if s + 1 < old.len() && !lesions.disable_down {
                    let parent = old[s + 1].contiguous()?;
                    let delta = self.down_delta.forward(&parent)?.tanh()?;
                    let gate = candle_nn::ops::sigmoid(&self.down_gate.forward(&parent)?)?;
                    let children = (delta * gate)?
                        .reshape((b, 2 * lengths[s + 1], c))?
                        .narrow(1, 0, lengths[s])?;
                    next = (&next + (children * self.config.exchange_step)?)?;
                }
            }
            result.push(next.clamp(-self.config.state_bound, self.config.state_bound)?);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn model(config: &SubstrateConfig) -> Result<(Substrate, VarMap)> {
        let vm = VarMap::new();
        let field = FieldConfig {
            seq_len: 17,
            channels: config.channels,
            periodic_boundary: false,
        };
        let nca = NcaConfig {
            hidden_dim: config.hidden_dim,
            ..Default::default()
        };
        let model = Substrate::new(
            VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu),
            config,
            &nca,
            &field,
        )?;
        crate::baselines::initialize_seeded(&vm, 42)?;
        Ok((model, vm))
    }

    fn small() -> SubstrateConfig {
        SubstrateConfig {
            channels: 8,
            hidden_dim: 12,
            scales: 8,
            ..Default::default()
        }
    }

    #[test]
    fn shapes_odd_lengths_and_zero_initial_coarse_states() -> Result<()> {
        let config = small();
        let (model, _) = model(&config)?;
        for l in [1, 3, 5, 17] {
            let initial = model.initialize(Tensor::ones((2, l, 8), DType::F32, &Device::Cpu)?)?;
            let states = model.states(&initial);
            assert_eq!(
                states.iter().map(|x| x.dim(1).unwrap()).collect::<Vec<_>>(),
                config.scale_lengths(l)?
            );
            for x in states.iter().skip(1) {
                assert_eq!(x.abs()?.sum_all()?.to_scalar::<f32>()?, 0.0);
            }
            let stepped = model.step(&initial, &Lesions::default())?;
            assert_eq!(model.fine(&stepped).dims(), &[2, l, 8]);
        }
        Ok(())
    }

    #[test]
    fn ballistic_pulses_no_wrap_both_directions_and_memory_stays() -> Result<()> {
        let config = SubstrateConfig {
            scales: 1,
            reaction_step: 0.0,
            exchange_step: 0.0,
            ..small()
        };
        let (model, _) = model(&config)?;
        let mut values = vec![0f32; 5 * 8];
        values[2 * 8] = 0.5; // memory
        values[2 * 8 + 2] = 1.0; // left
        values[2 * 8 + 4] = 1.0; // right
        let mut s = model.initialize(Tensor::from_vec(values, (1, 5, 8), &Device::Cpu)?)?;
        for tick in 1..=4 {
            s = model.step(&s, &Lesions::default())?;
            let values = model.fine(&s).to_vec3::<f32>()?;
            for (i, cell) in values[0].iter().enumerate() {
                assert_eq!(cell[0], if i == 2 { 0.5 } else { 0.0 });
                assert_eq!(cell[2], if tick <= 2 && i == 2 - tick { 1.0 } else { 0.0 });
                assert_eq!(cell[4], if tick <= 2 && i == 2 + tick { 1.0 } else { 0.0 });
            }
        }
        Ok(())
    }

    #[test]
    fn ordered_pair_mapping_retains_child_parity_and_pads_odd_tail() -> Result<()> {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 3, 2), &Device::Cpu)?;
        assert_eq!(
            ordered_pairs(&x)?.to_vec3::<f32>()?,
            vec![vec![vec![1., 2., 3., 4.], vec![5., 6., 0., 0.]]]
        );
        Ok(())
    }

    #[test]
    fn empirical_perturbations_respect_synchronous_geometry_and_use_short_route() -> Result<()> {
        let config = SubstrateConfig {
            scales: 4,
            ..small()
        };
        let (model, vm) = model(&config)?;
        // Positive weights ensure a measurable effect along every available edge.
        for (name, var) in vm.data().lock().unwrap().iter() {
            let value = if name.ends_with("bias") { 0.0 } else { 0.1 };
            var.set(&(var.ones_like()? * value)?)?;
        }
        let lengths = config.scale_lengths(17)?;
        let zero = Tensor::zeros((1, 17, 8), DType::F32, &Device::Cpu)?;
        let mut values = vec![0f32; 17 * 8];
        values[..8].fill(0.1);
        let mut control = model.initialize(zero)?;
        let mut pulse = model.initialize(Tensor::from_vec(values, (1, 17, 8), &Device::Cpu)?)?;
        let mut reached: Vec<Vec<bool>> = lengths.iter().map(|l| vec![false; *l]).collect();
        reached[0][0] = true;
        let mut endpoint_arrival = None;
        for tick in 1..=9 {
            let mut next = reached.clone();
            for s in 0..lengths.len() {
                for i in 0..lengths[s] {
                    if reached[s][i] {
                        if i > 0 {
                            next[s][i - 1] = true;
                        }
                        if i + 1 < lengths[s] {
                            next[s][i + 1] = true;
                        }
                        if s + 1 < lengths.len() {
                            next[s + 1][i / 2] = true;
                        }
                        if s > 0 {
                            next[s - 1][2 * i] = true;
                            if 2 * i + 1 < lengths[s - 1] {
                                next[s - 1][2 * i + 1] = true;
                            }
                        }
                    }
                }
            }
            reached = next;
            control = model.step(&control, &Lesions::default())?;
            pulse = model.step(&pulse, &Lesions::default())?;
            for (s, (a, b)) in model
                .states(&control)
                .iter()
                .zip(model.states(&pulse))
                .enumerate()
            {
                let diff = (*a - b)?.abs()?.to_vec3::<f32>()?;
                for i in 0..lengths[s] {
                    if !reached[s][i] {
                        assert!(
                            diff[0][i].iter().all(|v| *v == 0.0),
                            "unexpected shortcut tick={tick} scale={s} pos={i}"
                        );
                    }
                }
                if s == 0 && diff[0][16].iter().any(|v| *v > 1e-9) && endpoint_arrival.is_none() {
                    endpoint_arrival = Some(tick);
                }
            }
        }
        assert_eq!(endpoint_arrival, Some(8));
        // No upward exchange means fine perturbations cannot enter any coarse state.
        let a = model.initialize(Tensor::zeros((1, 17, 8), DType::F32, &Device::Cpu)?)?;
        let b = model.initialize(Tensor::ones((1, 17, 8), DType::F32, &Device::Cpu)?)?;
        let lesion = Lesions {
            disable_up: true,
            ..Default::default()
        };
        let a = model.evolve(&a, 4, &lesion)?;
        let b = model.evolve(&b, 4, &lesion)?;
        for (x, y) in model.states(&a).iter().zip(model.states(&b)).skip(1) {
            assert_eq!((*x - y)?.abs()?.sum_all()?.to_scalar::<f32>()?, 0.0);
        }
        Ok(())
    }

    #[test]
    fn finite_nonzero_gradients_reach_reaction_and_both_exchange_directions() -> Result<()> {
        let (model, vm) = model(&small())?;
        let input = Tensor::from_vec(
            (0..2 * 5 * 8).map(|i| (i as f32 * 0.31).sin()).collect(),
            (2, 5, 8),
            &Device::Cpu,
        )?;
        let state = model.evolve(&model.initialize(input)?, 4, &Lesions::default())?;
        let loss = model.fine(&state).sqr()?.mean_all()?;
        let grads = loss.backward()?;
        for (name, var) in vm.data().lock().unwrap().iter() {
            let grad = grads
                .get(var)
                .unwrap_or_else(|| panic!("missing gradient {name}"));
            let values = grad.flatten_all()?.to_vec1::<f32>()?;
            assert!(values.iter().all(|x| x.is_finite()), "nonfinite {name}");
            assert!(
                values.iter().any(|x| x.abs() > 1e-12),
                "zero gradient {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn deterministic_seed_config_and_weight_checkpoint_roundtrip() -> Result<()> {
        let config = small();
        let serialized = serde_json::to_string(&config)?;
        let decoded: SubstrateConfig = serde_json::from_str(&serialized)?;
        assert_eq!(config, decoded);
        let (a, vm) = model(&config)?;
        let (b, mut other) = model(&decoded)?;
        let input = Tensor::ones((2, 5, 8), DType::F32, &Device::Cpu)?;
        let output = |m: &Substrate| -> Result<Vec<f32>> {
            let s = m.evolve(&m.initialize(input.clone())?, 3, &Lesions::default())?;
            Ok(m.fine(&s).flatten_all()?.to_vec1::<f32>()?)
        };
        assert_eq!(output(&a)?, output(&b)?);
        let path = std::env::temp_dir().join(format!(
            "titan-substrate-{}-roundtrip.safetensors",
            std::process::id()
        ));
        vm.save(&path)?;
        crate::baselines::initialize_seeded(&other, 999)?;
        assert_ne!(output(&a)?, output(&b)?);
        other.load(&path)?;
        std::fs::remove_file(&path)?;
        assert_eq!(output(&a)?, output(&b)?);
        Ok(())
    }

    #[test]
    fn legacy_adapter_preserves_parameter_names_and_exact_trajectory() -> Result<()> {
        let config = SubstrateConfig {
            kind: SubstrateKind::LegacyNca,
            ..small()
        };
        let (adapter, vm) = model(&config)?;
        assert!(vm
            .data()
            .lock()
            .unwrap()
            .keys()
            .all(|k| k.starts_with("nca.")));
        let input = Tensor::ones((2, 5, 8), DType::F32, &Device::Cpu)?;
        let initial = adapter.initialize(input)?;
        let nca = match &adapter {
            Substrate::Legacy(nca) => nca,
            _ => unreachable!(),
        };
        let mut direct = match &initial {
            State::Legacy(s) => s.clone(),
            _ => unreachable!(),
        };
        let mut adapted = initial;
        for _ in 0..4 {
            direct = nca.step_field(&direct, &Device::Cpu)?;
            adapted = adapter.step(&adapted, &Lesions::default())?;
            assert_eq!(
                direct.x.flatten_all()?.to_vec1::<f32>()?,
                adapter.fine(&adapted).flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn strict_config_and_lesion_validation() -> Result<()> {
        let mut config = small();
        config.channels = 7;
        assert!(config.validate().is_err());
        config = small();
        config.boundary = "periodic".into();
        assert!(config.validate().is_err());
        config = small();
        config.exchange_step = f64::NAN;
        assert!(config.validate().is_err());
        let (model, _) = model(&small())?;
        let state = model.initialize(Tensor::zeros((1, 3, 8), DType::F32, &Device::Cpu)?)?;
        assert!(model
            .step(
                &state,
                &Lesions {
                    shuffle_scale: Some(0),
                    ..Default::default()
                }
            )
            .is_err());
        assert!(model
            .step(
                &state,
                &Lesions {
                    reset_scale: Some(99),
                    ..Default::default()
                }
            )
            .is_err());
        Ok(())
    }
}
