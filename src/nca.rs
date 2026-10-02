use crate::config::{FieldConfig, NcaConfig};
use crate::field::{MorphogenicField, PingPongField};
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

#[derive(Clone)]
pub struct NeuralCellularAutomaton {
    pub dense1: Linear,
    pub dense_delta: Linear,
    pub dense_gate: Linear,
    /// Relay-Fold Transport projections (Some only when carry_quantization = "fold").
    /// Gate: sigmoid map from [incoming_carry ; h1_act] to carry gates.
    /// Candidate: tanh map from [incoming_carry ; h1_act] to the folded carry.
    pub carry_fold_gate: Option<Linear>,
    pub carry_fold_cand: Option<Linear>,
    pub macro_dense1: Option<Linear>,
    pub macro_dense_delta: Option<Linear>,
    pub macro_inject: Option<Linear>,
    pub nca_cfg: NcaConfig,
    pub field_cfg: FieldConfig,
}

impl NeuralCellularAutomaton {
    /// Creates a clone of this NCA with an altered Navier-Stokes viscosity parameter
    pub fn with_viscosity(&self, viscosity: f32) -> Self {
        let mut cfg = self.nca_cfg.clone();
        cfg.viscosity = viscosity;
        Self {
            dense1: self.dense1.clone(),
            dense_delta: self.dense_delta.clone(),
            dense_gate: self.dense_gate.clone(),
            carry_fold_gate: self.carry_fold_gate.clone(),
            carry_fold_cand: self.carry_fold_cand.clone(),
            macro_dense1: self.macro_dense1.clone(),
            macro_dense_delta: self.macro_dense_delta.clone(),
            macro_inject: self.macro_inject.clone(),
            nca_cfg: cfg,
            field_cfg: self.field_cfg.clone(),
        }
    }
    pub fn new(vb: VarBuilder, nca_cfg: &NcaConfig, field_cfg: &FieldConfig) -> Result<Self> {
        // Local 1D perception: [Identity (C), Gradient (C), Laplacian (C)] = 3 * C
        // Macro hierarchy adds local coarsened modulation channel only under legacy perception coupling: + C_M
        // Macro recursive feedback adds global collective state: + C = 4 * C
        let is_hierarchy = nca_cfg.is_hierarchy();
        let macro_channels = if is_hierarchy { nca_cfg.macro_channels.max(1) } else { 0 };
        let is_perception_coupling = is_hierarchy && nca_cfg.macro_coupling == "perception";
        let mut in_channels = if is_hierarchy {
            if is_perception_coupling {
                field_cfg.channels * 3 + macro_channels
            } else {
                field_cfg.channels * 3
            }
        } else if nca_cfg.has_feedback() {
            field_cfg.channels * 4
        } else {
            field_cfg.channels * 3
        };
        if nca_cfg.coord_channel {
            in_channels += 1;
        }
        if nca_cfg.persistent_input {
            in_channels += field_cfg.channels;
        }

        let dense1 = linear(in_channels, nca_cfg.hidden_dim, vb.pp("dense1"))?;
        let dense_delta = linear(nca_cfg.hidden_dim, field_cfg.channels, vb.pp("dense_delta"))?;
        let dense_gate = linear(nca_cfg.hidden_dim, field_cfg.channels, vb.pp("dense_gate"))?;

        // Relay-Fold Transport (RFT) projections: only created for the "fold"
        // carry mode so all other configurations keep their exact var sets.
        let (carry_fold_gate, carry_fold_cand) = if nca_cfg.carry_fold_mode() {
            let c_carry = nca_cfg.carry_channels.min(field_cfg.channels);
            let fold_in = c_carry + nca_cfg.hidden_dim;
            (
                Some(linear(fold_in, c_carry, vb.pp("carry_fold_gate"))?),
                Some(linear(fold_in, c_carry, vb.pp("carry_fold_cand"))?),
            )
        } else {
            (None, None)
        };

        let (macro_dense1, macro_dense_delta, macro_inject) = if is_hierarchy {
            let micro_pool_dim = if nca_cfg.macro_downsampler == "walsh" && nca_cfg.macro_stride == 2 {
                field_cfg.channels * 2
            } else {
                field_cfg.channels
            };
            let m_in_channels = macro_channels * 3 + micro_pool_dim;
            let m_hidden = (nca_cfg.hidden_dim / 2).max(16);
            let md1 = linear(m_in_channels, m_hidden, vb.pp("macro_dense1"))?;
            let mdd = linear(m_hidden, macro_channels, vb.pp("macro_dense_delta"))?;
            let minj = if nca_cfg.macro_coupling == "state_derivative" {
                Some(linear(macro_channels, field_cfg.channels, vb.pp("macro_inject"))?)
            } else {
                None
            };
            (Some(md1), Some(mdd), minj)
        } else {
            (None, None, None)
        };

        Ok(Self {
            dense1,
            dense_delta,
            dense_gate,
            carry_fold_gate,
            carry_fold_cand,
            macro_dense1,
            macro_dense_delta,
            macro_inject,
            nca_cfg: nca_cfg.clone(),
            field_cfg: field_cfg.clone(),
        })
    }

    /// Parameter count breakdown across recurrent subsystems
    pub fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let p_dense1 = self.dense1.weight().elem_count() + self.dense1.bias().map_or(0, |b| b.elem_count());
        let p_delta = self.dense_delta.weight().elem_count() + self.dense_delta.bias().map_or(0, |b| b.elem_count());
        let p_gate = self.dense_gate.weight().elem_count() + self.dense_gate.bias().map_or(0, |b| b.elem_count());

        let mut breakdown = vec![
            ("spatial_perception_stencil".to_string(), 0), // fixed differential operator
            ("hidden_feature_mlp_dense1".to_string(), p_dense1),
            ("directional_update_dense_delta".to_string(), p_delta),
            ("adaptive_gate_dense_gate".to_string(), p_gate),
            ("viscous_dissipation_damping".to_string(), 0), // physical continuum operator
        ];
        if self.nca_cfg.is_hierarchy() {
            if let Some(ref md1) = self.macro_dense1 {
                let p = md1.weight().elem_count() + md1.bias().map_or(0, |b| b.elem_count());
                breakdown.push(("macro_hierarchy_mlp_dense1".to_string(), p));
            }
            if let Some(ref mdd) = self.macro_dense_delta {
                let p = mdd.weight().elem_count() + mdd.bias().map_or(0, |b| b.elem_count());
                breakdown.push(("macro_hierarchy_directional_delta".to_string(), p));
            }
            if let Some(ref minj) = self.macro_inject {
                let p = minj.weight().elem_count() + minj.bias().map_or(0, |b| b.elem_count());
                breakdown.push(("macro_state_derivative_inject".to_string(), p));
            }
            breakdown.push(("macro_local_highpass_modulation".to_string(), 0));
        } else if self.nca_cfg.has_feedback() {
            breakdown.push(("macro_recursive_feedback_loop".to_string(), 0));
        }
        if self.nca_cfg.coord_channel {
            breakdown.push(("spatial_coordinate_channel".to_string(), 0));
        }
        if self.nca_cfg.persistent_input {
            breakdown.push(("persistent_input_channel".to_string(), 0));
        }
        if self.nca_cfg.carry_channels > 0 {
            breakdown.push(("explicit_carry_register".to_string(), 0));
        }
        if let (Some(gate_proj), Some(cand_proj)) = (&self.carry_fold_gate, &self.carry_fold_cand) {
            let p = gate_proj.weight().elem_count()
                + gate_proj.bias().map_or(0, |b| b.elem_count())
                + cand_proj.weight().elem_count()
                + cand_proj.bias().map_or(0, |b| b.elem_count());
            breakdown.push(("carry_relay_fold_transport".to_string(), p));
        }
        breakdown
    }

    /// Generates static 1D spatial coordinate channel p_i in [-1, 1] for B sequences of length L
    pub fn generate_coordinates(b: usize, l: usize, mode: Option<&str>, device: &Device) -> Result<Tensor> {
        let mut coords = Vec::with_capacity(l);
        if l <= 1 {
            coords.push(0.0f32);
        } else {
            for i in 0..l {
                let p = 2.0 * (i as f32) / ((l - 1) as f32) - 1.0;
                coords.push(p);
            }
        }

        match mode.unwrap_or("intact") {
            "zeroed" => {
                for c in coords.iter_mut() {
                    *c = 0.0;
                }
            }
            "constant" => {
                for c in coords.iter_mut() {
                    *c = 0.5;
                }
            }
            "reversed" => {
                coords.reverse();
            }
            "shuffled" => {
                use rand::seq::SliceRandom;
                use rand::SeedableRng;
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                coords.shuffle(&mut rng);
            }
            "intact" | _ => {}
        }

        let mut batch_coords = Vec::with_capacity(b * l);
        for _ in 0..b {
            batch_coords.extend_from_slice(&coords);
        }
        Ok(Tensor::from_vec(batch_coords, (b, l, 1), device)?)
    }

    /// Circular spatial shift: rolls tensor [B, L, C] along spatial dimension L
    fn roll_spatial(tensor: &Tensor, shift: i32) -> Result<Tensor> {
        let (_, l, _) = tensor.dims3()?;
        let shift = ((shift % l as i32) + l as i32) as usize % l;
        if shift == 0 {
            return Ok(tensor.clone());
        }

        let left_part = tensor.narrow(1, l - shift, shift)?;
        let right_part = tensor.narrow(1, 0, l - shift)?;
        Ok(Tensor::cat(&[&left_part, &right_part], 1)?)
    }

    /// Computes spatial perception: [x, grad, laplacian] (symmetric) or [x, left, x - left] (causal DAG)
    pub fn perceive_with_feedback(&self, x: &Tensor, slow_state: Option<&Tensor>) -> Result<Tensor> {
        self.perceive_with_feedback_and_seed(x, slow_state, None)
    }

    /// Computes spatial perception with optional macro feedback and persistent initial input seed
    pub fn perceive_with_feedback_and_seed(
        &self,
        x: &Tensor,
        slow_state: Option<&Tensor>,
        seed: Option<&Tensor>,
    ) -> Result<Tensor> {
        let base_perc = if self.nca_cfg.causal_stencil {
            let left = self.left_neighbor(x)?;
            // Causal basis: [x_i, x_{i-1}, x_i - x_{i-1}]
            // Spans the 2-point causal stencil {i-1, i} with strictly lower-bidiagonal Jacobian.
            let diff = (x - &left)?;

            if self.nca_cfg.is_hierarchy() {
                if self.nca_cfg.macro_coupling == "perception" {
                    let (b, l, _) = x.dims3()?;
                    let s = self.nca_cfg.macro_stride.max(1);
                    let lm = (l / s).max(1);
                    let cm = self.nca_cfg.macro_channels.max(1);

                    let m = match slow_state {
                        Some(s_ten) if s_ten.dims3().map(|d| d.1 == lm && d.2 == cm).unwrap_or(false) => s_ten.clone(),
                        _ => Tensor::zeros((b, lm, cm), candle_core::DType::F32, x.device())?,
                    };

                    let m_left = self.left_neighbor(&m)?;
                    let w_macro = (&m - &m_left)?;

                    let w = if s > 1 {
                        w_macro.unsqueeze(2)?.repeat((1, 1, s, 1))?.reshape((b, lm * s, cm))?
                    } else {
                        w_macro
                    };
                    let w = if w.dim(1)? > l {
                        w.narrow(1, 0, l)?
                    } else {
                        w
                    };

                    let gamma = (self.nca_cfg.feedback_weight as f64).clamp(0.001, 0.10);
                    let scaled_w = (w * gamma)?;

                    Tensor::cat(&[x, &left, &diff, &scaled_w], 2)?
                } else {
                    Tensor::cat(&[x, &left, &diff], 2)?
                }
            } else if self.nca_cfg.has_feedback() {
                let (_, l, _) = x.dims3()?;
                let s = match slow_state {
                    Some(s_ten) => s_ten.clone(),
                    None => x.mean(1)?.unsqueeze(1)?,
                };
                let s_broadcast = s.repeat((1, l, 1))?;
                Tensor::cat(&[x, &left, &diff, &s_broadcast], 2)?
            } else {
                Tensor::cat(&[x, &left, &diff], 2)?
            }
        } else {
            let (left, right) = self.neighbors(x)?;

            // 1st spatial derivative: (right - left) / 2
            let grad = ((&right - &left)? / 2.0)?;

            // 2nd spatial derivative (Laplacian): right - 2 * x + left
            let double_x = (x * 2.0)?;
            let diff = (&right - &double_x)?;
            let laplacian = (&diff + &left)?;

            if self.nca_cfg.is_hierarchy() {
                if self.nca_cfg.macro_coupling == "perception" {
                    let (b, l, _) = x.dims3()?;
                    let s = self.nca_cfg.macro_stride.max(1);
                    let lm = (l / s).max(1);
                    let cm = self.nca_cfg.macro_channels.max(1);

                    let m = match slow_state {
                        Some(s_ten) if s_ten.dims3().map(|d| d.1 == lm && d.2 == cm).unwrap_or(false) => s_ten.clone(),
                        _ => Tensor::zeros((b, lm, cm), candle_core::DType::F32, x.device())?,
                    };

                    // Local discrete high-pass / difference stencil on macro grid:
                    // w_macro_j = M_j - 0.5 * (M_{j-1} + M_{j+1})
                    // Zero DC response guaranteed by construction!
                    let (m_left, m_right) = self.neighbors(&m)?;
                    let lap_avg = ((&m_left + &m_right)? * 0.5)?;
                    let w_macro = (&m - &lap_avg)?;

                    // Nearest-neighbor upsampling to micro resolution:
                    let w = if s > 1 {
                        w_macro.unsqueeze(2)?.repeat((1, 1, s, 1))?.reshape((b, lm * s, cm))?
                    } else {
                        w_macro.clone()
                    };
                    let w = if w.dim(1)? > l {
                        w.narrow(1, 0, l)?
                    } else {
                        w
                    };

                    // Hard-bounded coupling gain: gamma in [0.001, 0.10]
                    let gamma = (self.nca_cfg.feedback_weight as f64).clamp(0.001, 0.10);
                    let scaled_w = (w * gamma)?;

                    Tensor::cat(&[x, &grad, &laplacian, &scaled_w], 2)?
                } else {
                    // In state_derivative coupling mode, micro perception is strictly local
                    Tensor::cat(&[x, &grad, &laplacian], 2)?
                }
            } else if self.nca_cfg.has_feedback() {
                let (_, l, _) = x.dims3()?;
                let s = match slow_state {
                    Some(s_ten) => s_ten.clone(),
                    None => x.mean(1)?.unsqueeze(1)?,
                };
                let s_broadcast = s.repeat((1, l, 1))?;
                Tensor::cat(&[x, &grad, &laplacian, &s_broadcast], 2)?
            } else {
                // Concatenate along channels: [B, L, 3 * C]
                Tensor::cat(&[x, &grad, &laplacian], 2)?
            }
        };

        let perc = if self.nca_cfg.coord_channel {
            let (b, l, _) = x.dims3()?;
            let mode = self.nca_cfg.coord_train_mode.as_deref();
            let coords = Self::generate_coordinates(b, l, mode, x.device())?;
            Tensor::cat(&[&base_perc, &coords], 2)?
        } else {
            base_perc
        };

        if self.nca_cfg.persistent_input {
            let s = match seed {
                Some(seed_ten) => seed_ten.clone(),
                None => x.clone(),
            };
            Ok(Tensor::cat(&[&perc, &s], 2)?)
        } else {
            Ok(perc)
        }
    }

    /// Computes local 1D perception: Identity, 1st Derivative (Gradient), 2nd Derivative (Laplacian)
    pub fn perceive(&self, x: &Tensor) -> Result<Tensor> {
        self.perceive_with_feedback(x, None)
    }

    /// Zero boundaries prevent end-to-start shortcuts when requested.
    fn neighbors(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.field_cfg.periodic_boundary {
            return Ok((Self::roll_spatial(x, 1)?, Self::roll_spatial(x, -1)?));
        }
        let (b, l, c) = x.dims3()?;
        let zero = Tensor::zeros((b, 1, c), x.dtype(), x.device())?;
        if l == 1 { return Ok((zero.clone(), zero)); }
        Ok((Tensor::cat(&[&zero, &x.narrow(1, 0, l - 1)?], 1)?,
            Tensor::cat(&[&x.narrow(1, 1, l - 1)?, &zero], 1)?))
    }

    /// Causal neighbor: strictly left neighbor x_{i-1} with zero Dirichlet boundary at i=0 (DAG fold)
    pub fn left_neighbor(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let zero = Tensor::zeros((b, 1, c), x.dtype(), x.device())?;
        if l == 1 { return Ok(zero); }
        Ok(Tensor::cat(&[&zero, &x.narrow(1, 0, l - 1)?], 1)?)
    }

    /// Anti-causal neighbor: strictly right neighbor x_{i+1} with zero Dirichlet boundary at i=L-1
    pub fn right_neighbor(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let zero = Tensor::zeros((b, 1, c), x.dtype(), x.device())?;
        if l == 1 { return Ok(zero); }
        Ok(Tensor::cat(&[&x.narrow(1, 1, l - 1)?, &zero], 1)?)
    }

    /// Multi-hop left skip neighbor: x_{i-k} with zero Dirichlet boundary at i < k
    pub fn k_left_neighbor(&self, x: &Tensor, k: usize) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        if k >= l {
            return Ok(Tensor::zeros((b, l, c), x.dtype(), x.device())?);
        }
        let zero = Tensor::zeros((b, k, c), x.dtype(), x.device())?;
        Ok(Tensor::cat(&[&zero, &x.narrow(1, 0, l - k)?], 1)?)
    }

    /// Multi-hop right skip neighbor: x_{i+k} with zero Dirichlet boundary at i >= L-k
    pub fn k_right_neighbor(&self, x: &Tensor, k: usize) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        if k >= l {
            return Ok(Tensor::zeros((b, l, c), x.dtype(), x.device())?);
        }
        let zero = Tensor::zeros((b, k, c), x.dtype(), x.device())?;
        Ok(Tensor::cat(&[&x.narrow(1, k, l - k)?, &zero], 1)?)
    }

    /// Multi-hop left skip neighbor with CLAMPED launch: x_{max(i-k, 0)}.
    /// Cells 0..k-1 read the boundary cell x_0 instead of zero, so the first
    /// cell can always launch and a value seeded at cell 0 reaches every cell q
    /// in ceil(q/k) ticks (as opposed to the legacy zero-pad `k_left_neighbor`,
    /// which reaches only q ≡ 0 (mod k) through the k-hop path). Opt-in via
    /// `carry_launch_clamp`, always on for Relay-Fold Transport ("fold").
    pub fn k_left_neighbor_clamped(&self, x: &Tensor, k: usize) -> Result<Tensor> {
        let (_, l, _) = x.dims3()?;
        let kk = k.min(l);
        // out[0..kk] = x_0, out[kk..L] = x[0..L-kk]
        let first = x.narrow(1, 0, 1)?.repeat((1, kk, 1))?;
        if kk >= l {
            return Ok(first);
        }
        let tail = x.narrow(1, 0, l - kk)?;
        Ok(Tensor::cat(&[&first, &tail], 1)?)
    }

    /// Multi-hop right skip neighbor with CLAMPED launch: x_{min(i+k, L-1)}.
    /// Symmetric to [`Self::k_left_neighbor_clamped`]: cells L-k..L-1 read the
    /// boundary cell x_{L-1} instead of zero.
    pub fn k_right_neighbor_clamped(&self, x: &Tensor, k: usize) -> Result<Tensor> {
        let (_, l, _) = x.dims3()?;
        let kk = k.min(l);
        // out[0..L-kk] = x[kk..L], out[L-kk..L] = x_{L-1}
        let last = x.narrow(1, l - 1, 1)?.repeat((1, kk, 1))?;
        if kk >= l {
            return Ok(last);
        }
        let head = x.narrow(1, kk, l - kk)?;
        Ok(Tensor::cat(&[&head, &last], 1)?)
    }

    /// Explicit Carry Register transport: hyperbolic upwind advection of the
    /// carry channels by up to k cells per tick.
    ///
    /// - Legacy modes (every carry_quantization value except "fold") keep the
    ///   historical slow/fast half-channel split byte-identical: with k > 1,
    ///   half the carry channels hop 1 cell/tick and half hop k cells/tick,
    ///   zero-padded at the borders. `carry_launch_clamp` optionally switches
    ///   the launch boundary from zero-pad to clamp (off by default).
    /// - Relay-Fold Transport ("fold") applies k uniformly to ALL carry
    ///   channels (a k sweep is a real speed knob) and clamps the launch so the
    ///   first/last k cells can relay.
    pub fn shift_carry(&self, c: &Tensor) -> Result<Tensor> {
        let k = self.nca_cfg.carry_skip_stride.max(1);
        let uniform = self.nca_cfg.carry_uniform_stride();
        let clamp = self.nca_cfg.carry_clamped_launch();

        let left1 = |t: &Tensor| -> Result<Tensor> {
            if clamp { self.k_left_neighbor_clamped(t, 1) } else { self.left_neighbor(t) }
        };
        let leftk = |t: &Tensor| -> Result<Tensor> {
            if clamp { self.k_left_neighbor_clamped(t, k) } else { self.k_left_neighbor(t, k) }
        };
        let right1 = |t: &Tensor| -> Result<Tensor> {
            if clamp { self.k_right_neighbor_clamped(t, 1) } else { self.right_neighbor(t) }
        };
        let rightk = |t: &Tensor| -> Result<Tensor> {
            if clamp { self.k_right_neighbor_clamped(t, k) } else { self.k_right_neighbor(t, k) }
        };

        let (_, _, cn) = c.dims3()?;
        if uniform {
            // RFT: speed k on every carry channel.
            if self.nca_cfg.carry_bidirectional {
                let half = cn / 2;
                let c_fwd = c.narrow(2, 0, half)?;
                let c_bwd = c.narrow(2, half, cn - half)?;
                let fwd_shifted = leftk(&c_fwd)?;
                let bwd_shifted = rightk(&c_bwd)?;
                Ok(Tensor::cat(&[&fwd_shifted, &bwd_shifted], 2)?)
            } else {
                leftk(c)
            }
        } else if self.nca_cfg.carry_bidirectional {
            // Legacy bidirectional slow/fast split.
            let half = cn / 2;
            let c_fwd = c.narrow(2, 0, half)?;
            let c_bwd = c.narrow(2, half, cn - half)?;
            let fwd_shifted = if k > 1 {
                let half_fwd = half / 2;
                if half_fwd > 0 {
                    let c_fwd_slow = c_fwd.narrow(2, 0, half - half_fwd)?;
                    let c_fwd_fast = c_fwd.narrow(2, half - half_fwd, half_fwd)?;
                    let slow_shift = left1(&c_fwd_slow)?;
                    let fast_shift = leftk(&c_fwd_fast)?;
                    Tensor::cat(&[&slow_shift, &fast_shift], 2)?
                } else {
                    left1(&c_fwd)?
                }
            } else {
                left1(&c_fwd)?
            };
            let bwd_shifted = if k > 1 {
                let bwd_len = cn - half;
                let half_bwd = bwd_len / 2;
                if half_bwd > 0 {
                    let c_bwd_slow = c_bwd.narrow(2, 0, bwd_len - half_bwd)?;
                    let c_bwd_fast = c_bwd.narrow(2, bwd_len - half_bwd, half_bwd)?;
                    let slow_shift = right1(&c_bwd_slow)?;
                    let fast_shift = rightk(&c_bwd_fast)?;
                    Tensor::cat(&[&slow_shift, &fast_shift], 2)?
                } else {
                    right1(&c_bwd)?
                }
            } else {
                right1(&c_bwd)?
            };
            Ok(Tensor::cat(&[&fwd_shifted, &bwd_shifted], 2)?)
        } else if k > 1 {
            // Legacy unidirectional slow/fast split.
            let half = cn / 2;
            if half > 0 {
                let c_slow = c.narrow(2, 0, cn - half)?;
                let c_fast = c.narrow(2, cn - half, half)?;
                let slow_shift = left1(&c_slow)?;
                let fast_shift = leftk(&c_fast)?;
                Ok(Tensor::cat(&[&slow_shift, &fast_shift], 2)?)
            } else {
                left1(c)
            }
        } else {
            left1(c)
        }
    }

    /// Discrete carry projection / drift mitigation to preserve clean discrete signals over deep horizons (L >= 64).
    /// - "none": standard continuous floating-point propagation.
    /// - "ste_round": HISTORICAL zero-gradient STE: `carry + (round(carry).detach() - carry)`.
    ///   Forward = integer rounding, but dy/dcarry == 0, so carry channels receive
    ///   NO gradient. Kept byte-identical for old checkpoints; not usable for learning.
    /// - "ste_sign": HISTORICAL zero-gradient STE for sign {-1, 0, 1}; same caveat.
    /// - "ste_round_ide": identity-gradient STE: `carry + (round(carry) - carry).detach()`.
    ///   Forward = integer rounding, dy/dcarry == 1 (trainable).
    /// - "ste_sign_ide": identity-gradient STE for sign {-1, 0, 1}, dy/dcarry == 1.
    /// - "bistable": continuous cubic Ginzburg-Landau restoring potential: c + 0.1 * c * (1 - c^2).
    /// - "fold": NOT a post-hoc projection: Relay-Fold Transport is a carry WRITE
    ///   mode applied inside `step_with_forcing`; calling this directly is an error.
    pub fn quantize_carry(carry: &Tensor, mode: &str) -> Result<Tensor> {
        match mode {
            "none" => Ok(carry.clone()),
            "ste_round" => {
                let rounded = carry.round()?;
                let diff = (rounded.detach() - carry)?;
                Ok((carry + diff)?)
            }
            "ste_sign" => {
                let pos = carry.gt(0.0)?.to_dtype(carry.dtype())?;
                let neg = carry.lt(0.0)?.to_dtype(carry.dtype())?;
                let sign = (&pos - &neg)?;
                let diff = (sign.detach() - carry)?;
                Ok((carry + diff)?)
            }
            "ste_round_ide" => {
                let rounded = carry.round()?;
                let diff = (rounded - carry)?.detach();
                Ok((carry + diff)?)
            }
            "ste_sign_ide" => {
                let pos = carry.gt(0.0)?.to_dtype(carry.dtype())?;
                let neg = carry.lt(0.0)?.to_dtype(carry.dtype())?;
                let sign = (&pos - &neg)?;
                let diff = (sign - carry)?.detach();
                Ok((carry + diff)?)
            }
            "bistable" => {
                let one = Tensor::ones_like(carry)?;
                let one_minus_c2 = (one - carry.sqr()?)?;
                let restoring = carry.mul(&one_minus_c2)?;
                Ok((carry + (restoring * 0.1)?)?)
            }
            "fold" => anyhow::bail!(
                "'fold' is a Relay-Fold Transport write mode applied inside step_with_forcing, not a post-hoc carry projection"
            ),
            other => anyhow::bail!("unknown carry_quantization mode '{}'", other),
        }
    }

    /// Relay-Fold Transport write for the Explicit Carry Register.
    ///
    /// At every relay cell the carry channel is re-written from the incoming
    /// transported carry c_{i-k} and the cell's own perception-derived hidden
    /// feature h_i (which itself reads the left neighbor, i.e. the incoming
    /// carry, through the causal stencil):
    ///
    ///   g   = sigmoid(W_g . [c_{i-k} ; h_i] + b_g)
    ///   z   = tanh(W_c . [c_{i-k} ; h_i] + b_c)
    ///   c_i = g (*) c_{i-k} + (1 - g) (*) z
    ///
    /// The candidate z is a NONLINEAR function of the incoming carry AND the
    /// local content jointly, so the accumulated value is re-folded at each hop
    /// (accumulator semantics: prefix combine at every relay), instead of being
    /// copied and additively perturbed. Local + translation-equivariant
    /// (shared projections, elementwise gating).
    pub fn carry_fold_write(&self, incoming: &Tensor, h_act: &Tensor) -> Result<Tensor> {
        match (&self.carry_fold_gate, &self.carry_fold_cand) {
            (Some(gate_proj), Some(cand_proj)) => {
                let feats = Tensor::cat(&[incoming, h_act], 2)?;
                let gate = candle_nn::ops::sigmoid(&gate_proj.forward(&feats)?)?;
                let cand = cand_proj.forward(&feats)?.tanh()?;
                let keep = gate.mul(incoming)?;
                let one = Tensor::ones_like(&gate)?;
                let refresh = (one - &gate)?.mul(&cand)?;
                Ok((keep + refresh)?)
            }
            _ => anyhow::bail!(
                "relay fold write requested but fold projections are not initialized \
                 (carry_quantization='fold' requires carry_channels > 0 at model construction)"
            ),
        }
    }


    pub fn dissipate(&self, tensor: &Tensor, effective_diff: f32) -> Result<Tensor> {
        if self.field_cfg.periodic_boundary && !self.nca_cfg.causal_stencil {
            return Self::apply_viscous_dissipation(tensor, effective_diff);
        }
        if effective_diff <= 0.0 { return Ok(tensor.clone()); }
        let n = (effective_diff / 0.25).ceil().max(1.0) as usize;
        let mut x = tensor.clone();
        for _ in 0..n {
            if self.nca_cfg.causal_stencil {
                let left = self.left_neighbor(&x)?;
                let upwind_diff = (&left - &x)?;
                x = (&x + (upwind_diff * (effective_diff as f64 / n as f64))?)?;
            } else {
                let (left, right) = self.neighbors(&x)?;
                let lap = ((left + right)? - (&x * 2.0)?)?;
                x = (&x + (lap * (effective_diff as f64 / n as f64))?)?;
            }
        }
        Ok(x)
    }

    /// Applies physical Navier-Stokes viscous dissipation (\nu * \Delta x * \Delta t)
    /// with adaptive substepping to guarantee unconditional von Neumann stability:
    /// For 1D central Laplacian stencil [1, -2, 1], maximum stable substep is D_max = 0.25.
    /// If total diffusion weight D = \nu * \alpha > 0.25, substepping partitions D into N = ceil(D / 0.25)
    /// substeps of size D / N <= 0.25. This ensures strictly dissipative, monotonic damping
    /// without artificial high-frequency numerical oscillations or explosions for any \nu >= 0.
    pub fn apply_viscous_dissipation(tensor: &Tensor, effective_diff: f32) -> Result<Tensor> {
        if effective_diff <= 0.0 {
            return Ok(tensor.clone());
        }
        let max_substep = 0.25f32;
        let num_substeps = (effective_diff / max_substep).ceil().max(1.0) as usize;
        let dt_sub = (effective_diff / num_substeps as f32) as f64;

        let mut current = tensor.clone();
        for _ in 0..num_substeps {
            let left = MorphogenicField::roll_spatial(&current, 1)?;
            let right = MorphogenicField::roll_spatial(&current, -1)?;
            let double = (&current * 2.0)?;
            let lap = (&right - &double)?;
            let lap = (&lap + &left)?;
            let damping = (lap * dt_sub)?;
            current = (&current + &damping)?;
        }
        Ok(current)
    }

    /// Projects cell states onto a compact invariant manifold:
    /// - "rms": Zero-parameter RMS normalization along channel dimension
    /// - "bounded": Smoothly clamps RMS amplitude above bound_threshold while leaving normal dynamics untouched
    /// - "layer_norm": Zero-mean, unit-variance LayerNorm along channel dimension
    #[allow(dead_code)]
    pub fn apply_state_norm(tensor: &Tensor, mode: &str) -> Result<Tensor> {
        Self::apply_state_norm_with_threshold(tensor, mode, 1.5)
    }

    pub fn apply_state_norm_with_threshold(tensor: &Tensor, mode: &str, threshold: f32) -> Result<Tensor> {
        match mode {
            "rms" => {
                let sq = tensor.sqr()?;
                let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
                let rms = (mean_sq + 1e-5)?.sqrt()?;
                Ok(tensor.broadcast_div(&rms)?)
            }
            "bounded" => {
                let sq = tensor.sqr()?;
                let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
                let rms = (mean_sq + 1e-5)?.sqrt()?;
                let th = (threshold as f64).max(0.1);
                let ratio = (rms / th)?;
                let ones = ratio.ones_like()?;
                let scale = ratio.maximum(&ones)?;
                Ok(tensor.broadcast_div(&scale)?)
            }
            "layer_norm" => {
                let mean = tensor.mean_keepdim(candle_core::D::Minus1)?;
                let centered = tensor.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let std = (var + 1e-5)?.sqrt()?;
                Ok(centered.broadcast_div(&std)?)
            }
            _ => Ok(tensor.clone()),
        }
    }

    /// Single developmental step with optional external forcing:
    /// \partial_t x = N_\theta(x) + \nu \Delta x + f(s, t)
    /// Discretized with time step \Delta t = \alpha = step_size
    /// Returns (new_field, update_norm)
    pub fn step_with_forcing(
        &self,
        field: &MorphogenicField,
        forcing: Option<&Tensor>,
        device: &Device,
    ) -> Result<(MorphogenicField, f32)> {
        let x = &field.x;

        // 1. Spatial perception with optional recursive macro feedback and persistent input
        let perception = self.perceive_with_feedback_and_seed(
            x,
            field.slow_state.as_ref(),
            field.seed.as_ref(),
        )?;

        // 2. Hidden feature extraction
        let h1 = self.dense1.forward(&perception)?;
        let h1_act = match self.nca_cfg.activation.as_str() {
            "tanh" => h1.tanh()?,
            _ => candle_nn::Activation::Gelu.forward(&h1)?,
        };

        // 3. Gated residual update: delta in (-1, 1), gate in (0, 1)
        let delta = self.dense_delta.forward(&h1_act)?.tanh()?;
        let gate = candle_nn::ops::sigmoid(&self.dense_gate.forward(&h1_act)?)?;
        let gated_delta = delta.mul(&gate)?;

        // 4. Optional stochastic update mask (for asynchronous cellular dynamics)
        let effective_delta = if self.nca_cfg.update_rate < 0.999 {
            let (b, l, c) = x.dims3()?;
            let mask_vals: Vec<f32> = (0..(b * l))
                .map(|_| if rand::random::<f32>() < self.nca_cfg.update_rate { 1.0 } else { 0.0 })
                .collect();
            let mask = Tensor::from_slice(&mask_vals, (b, l, 1), device)?
                .repeat((1, 1, c))?;
            gated_delta.mul(&mask)?
        } else {
            gated_delta
        };

        // 5. Residual active drift integration with optional damping, leaky contraction, and Explicit Carry Register (ECR):
        // For stationary channels: base_h = (1 - lambda) * h_i^t
        // For carry channels: base_c = transport(c)_i (hyperbolic upwind advection shift;
        // see shift_carry for legacy slow/fast split vs Relay-Fold Transport semantics)
        let alpha = (self.nca_cfg.step_size * self.nca_cfg.damping_alpha) as f64;
        let scaled_delta = (effective_delta * alpha)?;
        let base_x = if self.nca_cfg.carry_channels > 0 {
            let (_, _, c_total) = x.dims3()?;
            let c_carry = self.nca_cfg.carry_channels.min(c_total);
            let c_hidden = c_total - c_carry;
            let h = x.narrow(2, 0, c_hidden)?;
            let c = x.narrow(2, c_hidden, c_carry)?;
            let shifted_c = self.shift_carry(&c)?;

            let base_h = if self.nca_cfg.leaky_lambda > 0.0 {
                (h * (1.0 - self.nca_cfg.leaky_lambda.clamp(0.0, 0.99) as f64))?
            } else {
                h
            };
            Tensor::cat(&[&base_h, &shifted_c], 2)?
        } else if self.nca_cfg.leaky_lambda > 0.0 {
            (x * (1.0 - self.nca_cfg.leaky_lambda.clamp(0.0, 0.99) as f64))?
        } else {
            x.clone()
        };
        let mut interim_x = (&base_x + &scaled_delta)?;

        // 5b. Relay-Fold Transport write (carry_quantization = "fold"):
        // The carry channels are NOT advected + additively perturbed; they are
        // re-written at every cell as a gated fold of the incoming transported
        // carry and the local perception-derived feature h1_act (see
        // carry_fold_write). The additive delta path is hidden-only under RFT.
        if self.nca_cfg.carry_fold_mode() {
            let (_, _, c_total) = interim_x.dims3()?;
            let c_carry = self.nca_cfg.carry_channels.min(c_total);
            let c_hidden = c_total - c_carry;
            let h_new = interim_x.narrow(2, 0, c_hidden)?;
            let c_in = x.narrow(2, c_hidden, c_carry)?;
            let incoming = self.shift_carry(&c_in)?;
            let folded = self.carry_fold_write(&incoming, &h1_act)?;
            interim_x = Tensor::cat(&[&h_new, &folded], 2)?;
        }

        // State-derivative macro injection (Track 1 / Path A):
        if self.nca_cfg.is_hierarchy() && self.nca_cfg.macro_coupling == "state_derivative" {
            if let Some(ref minj) = self.macro_inject {
                let (b, l, _) = x.dims3()?;
                let s = self.nca_cfg.macro_stride.max(1);
                let lm = (l / s).max(1);
                let cm = self.nca_cfg.macro_channels.max(1);

                let m = match &field.slow_state {
                    Some(s_ten) if s_ten.dims3().map(|d| d.1 == lm && d.2 == cm).unwrap_or(false) => s_ten.clone(),
                    _ => Tensor::zeros((b, lm, cm), candle_core::DType::F32, device)?,
                };

                let w_macro = if self.nca_cfg.causal_stencil {
                    let m_left = self.left_neighbor(&m)?;
                    (&m - &m_left)?
                } else {
                    let (m_left, m_right) = self.neighbors(&m)?;
                    let lap_avg = ((&m_left + &m_right)? * 0.5)?;
                    (&m - &lap_avg)?
                };

                let w = if s > 1 {
                    w_macro.unsqueeze(2)?.repeat((1, 1, s, 1))?.reshape((b, lm * s, cm))?
                } else {
                    w_macro
                };
                let w = if w.dim(1)? > l {
                    w.narrow(1, 0, l)?
                } else {
                    w
                };

                let gamma = (self.nca_cfg.macro_gamma as f64).clamp(0.001, 1.0);
                let inj_h = minj.forward(&w)?.tanh()?;
                let one_minus_x2 = (Tensor::ones_like(x)? - x.sqr()?)?.clamp(0.0, 1.0)?;
                let gated_inj = (inj_h * gamma)?.mul(&one_minus_x2)?;
                let scaled_inj = (gated_inj * alpha)?;
                interim_x = (&interim_x + &scaled_inj)?;
            }
        }

        if let Some(f) = forcing {
            let scaled_f = (f * alpha)?;
            interim_x = (&interim_x + &scaled_f)?;
        }

        // 6. State normalization: projects onto compact invariant manifold (prevents open-phase energy explosion)
        let interim_x = match self.nca_cfg.state_norm.as_str() {
            "bounded_h_only" => {
                if self.nca_cfg.carry_channels > 0 {
                    let (_, _, c_total) = interim_x.dims3()?;
                    let c_carry = self.nca_cfg.carry_channels.min(c_total);
                    let c_hidden = c_total - c_carry;
                    let h = interim_x.narrow(2, 0, c_hidden)?;
                    let c = interim_x.narrow(2, c_hidden, c_carry)?;
                    let norm_h = Self::apply_state_norm_with_threshold(&h, "bounded", self.nca_cfg.bound_threshold)?;
                    Tensor::cat(&[&norm_h, &c], 2)?
                } else {
                    Self::apply_state_norm_with_threshold(&interim_x, "bounded", self.nca_cfg.bound_threshold)?
                }
            }
            "none" => interim_x,
            other => Self::apply_state_norm_with_threshold(&interim_x, other, self.nca_cfg.bound_threshold)?,
        };

        // 7. Navier-Stokes physical viscous dissipation: \nu * \Delta x * \alpha
        // Scales consistently with time step alpha and employs adaptive substepping
        let mut new_x = if self.nca_cfg.viscosity > 0.0 {
            let effective_diff = self.nca_cfg.viscosity.max(0.0) * self.nca_cfg.step_size;
            self.dissipate(&interim_x, effective_diff)?
        } else {
            interim_x
        };

        // 7b. Carry register quantization / drift mitigation
        // (skipped under Relay-Fold Transport: the fold write is already a
        // bounded tanh re-encoding, not an advective copy to project)
        if self.nca_cfg.carry_channels > 0
            && self.nca_cfg.carry_quantization != "none"
            && !self.nca_cfg.carry_fold_mode()
        {
            let (_, _, c_total) = new_x.dims3()?;
            let c_carry = self.nca_cfg.carry_channels.min(c_total);
            let c_hidden = c_total - c_carry;
            let h = new_x.narrow(2, 0, c_hidden)?;
            let c = new_x.narrow(2, c_hidden, c_carry)?;
            let c_quantized = Self::quantize_carry(&c, &self.nca_cfg.carry_quantization)?;
            new_x = Tensor::cat(&[&h, &c_quantized], 2)?;
        }

        // 8. Macro hierarchy or recursive feedback loop:
        let new_slow_state = if self.nca_cfg.is_hierarchy() {
            let (b, l, c) = new_x.dims3()?;
            let s = self.nca_cfg.macro_stride.max(1);
            let lm = (l / s).max(1);
            let cm = self.nca_cfg.macro_channels.max(1);

            let prev_m = match &field.slow_state {
                Some(st) if st.dims3().map(|d| d.1 == lm && d.2 == cm).unwrap_or(false) => st.clone(),
                _ => Tensor::zeros((b, lm, cm), candle_core::DType::F32, device)?,
            };

            let is_macro_tick = (field.tick + 1) % self.nca_cfg.macro_period.max(1) == 0;
            if is_macro_tick {
                let downsampled_micro = if self.nca_cfg.macro_downsampler == "walsh" && s == 2 && l >= lm * 2 {
                    let truncated_x = new_x.narrow(1, 0, lm * 2)?;
                    let reshaped = truncated_x.reshape((b, lm, 2, c))?;
                    let x_even = reshaped.narrow(2, 0, 1)?.squeeze(2)?;
                    let x_odd = reshaped.narrow(2, 1, 1)?.squeeze(2)?;
                    let p_sum = (&x_even + &x_odd)?;
                    let p_diff = (&x_even - &x_odd)?;
                    Tensor::cat(&[&p_sum, &p_diff], 2)?
                } else if s > 1 && l >= lm * s {
                    let truncated_x = new_x.narrow(1, 0, lm * s)?;
                    truncated_x.reshape((b, lm, s, c))?.mean(candle_core::D::Minus2)?
                } else {
                    new_x.narrow(1, 0, lm)?
                };

                if let (Some(md1), Some(mdd)) = (&self.macro_dense1, &self.macro_dense_delta) {
                    let macro_perc = if self.nca_cfg.causal_stencil {
                        let m_left = self.left_neighbor(&prev_m)?;
                        let m_diff = (&prev_m - &m_left)?;
                        Tensor::cat(&[&m_left, &prev_m, &m_diff, &downsampled_micro], 2)?
                    } else {
                        let (m_left, m_right) = self.neighbors(&prev_m)?;
                        Tensor::cat(&[&m_left, &prev_m, &m_right, &downsampled_micro], 2)?
                    };
                    let mh1 = md1.forward(&macro_perc)?.tanh()?;
                    let m_delta = mdd.forward(&mh1)?.tanh()?;
                    let m_alpha = (self.nca_cfg.step_size * 0.5) as f64;
                    let scaled_md = (m_delta * m_alpha)?;
                    let mut updated_m = (&prev_m + &scaled_md)?;

                    // Emergent bistable potential: V(w) = (lambda / 4) * (w^2 - 1)^2
                    // Restoring force: -dV/dw = lambda * w * (1 - w^2)
                    if self.nca_cfg.macro_lambda > 0.0 {
                        let lambda = (self.nca_cfg.macro_lambda as f64).clamp(0.0, 1.0);
                        let one_minus_m2 = (Tensor::ones_like(&prev_m)? - prev_m.sqr()?)?;
                        let bistable_drift = prev_m.mul(&one_minus_m2)?;
                        let scaled_bistable = (bistable_drift * (lambda * m_alpha))?;
                        updated_m = (&updated_m + &scaled_bistable)?;
                    }

                    Some(updated_m)
                } else {
                    Some(prev_m)
                }
            } else {
                Some(prev_m)
            }
        } else if self.nca_cfg.has_feedback() {
            let pool_x = new_x.mean(1)?.unsqueeze(1)?; // [B, 1, C]
            if self.nca_cfg.feedback_mode == "dual_timescale" {
                let beta = (self.nca_cfg.feedback_weight as f64).clamp(0.01, 1.0);
                match &field.slow_state {
                    Some(prev_s) => {
                        let decay = (prev_s * (1.0 - beta))?;
                        let input = (&pool_x * beta)?;
                        Some((&decay + &input)?)
                    }
                    None => Some(pool_x),
                }
            } else {
                // "global_pool" mode: instantaneous spatial mean
                Some(pool_x)
            }
        } else {
            None
        };

        // Compute displacement norm ||x_(t+1) - x_t||
        let total_displacement = (&new_x - x)?;
        let update_sq = total_displacement.sqr()?;
        let update_norm = update_sq.mean_all()?.to_scalar::<f32>()?.sqrt();

        Ok((
            MorphogenicField {
                x: new_x,
                config: self.field_cfg.clone(),
                slow_state: new_slow_state,
                seed: field.seed.clone(),
                tick: field.tick + 1,
            },
            update_norm,
        ))
    }

    /// Single autonomous developmental step (unforced)
    pub fn step(&self, field: &MorphogenicField, device: &Device) -> Result<(MorphogenicField, f32)> {
        self.step_with_forcing(field, None, device)
    }

    /// Single step returning just field
    pub fn step_field(&self, field: &MorphogenicField, device: &Device) -> Result<MorphogenicField> {
        let (next_field, _) = self.step(field, device)?;
        Ok(next_field)
    }

    /// Unrolls development autonomously for T steps
    pub fn develop(&self, initial: &MorphogenicField, steps: usize, device: &Device) -> Result<MorphogenicField> {
        self.develop_with_intervention(initial, steps, None, device)
    }

    /// Unrolls development autonomously for T steps with optional controlled lesions/interventions
    pub fn develop_with_intervention(
        &self,
        initial: &MorphogenicField,
        steps: usize,
        intervention: Option<&crate::intervention::InterventionConfig>,
        device: &Device,
    ) -> Result<MorphogenicField> {
        let mut current = initial.clone();
        for s in 0..steps {
            if let Some(interv) = intervention {
                if interv.is_active() {
                    current.x = interv.apply_to_state(s, &current.x, device)?;
                }
            }
            current = self.step_field(&current, device)?;
        }
        if let Some(interv) = intervention {
            if interv.is_active() {
                current.x = interv.apply_to_state(steps, &current.x, device)?;
            }
        }
        Ok(current)
    }

    /// Zero-heap double-buffered development for ARM64 inference and benchmarking.
    /// Bit-for-bit mathematically equivalent to `develop`.
    #[allow(dead_code)]
    pub fn develop_ping_pong(
        &self,
        pp: &mut PingPongField,
        steps: usize,
        device: &Device,
    ) -> Result<()> {
        let b = pp.batch_size;
        let l = pp.seq_len;
        let c = pp.channels;
        anyhow::ensure!(c == self.field_cfg.channels, "ping-pong channels mismatch");

        // Fast path for standard causal stencil without macro hierarchy / feedback / viscosity / spatial norm
        // Relay-Fold Transport, uniform stride, and clamped launch are not
        // modeled by the fused row-walk; they fall back to step_field.
        let can_use_fast_fused = self.nca_cfg.causal_stencil
            && !self.nca_cfg.is_hierarchy()
            && !self.nca_cfg.has_feedback()
            && !self.nca_cfg.coord_channel
            && !self.nca_cfg.persistent_input
            && self.nca_cfg.state_norm == "none"
            && self.nca_cfg.viscosity <= 0.0
            && self.nca_cfg.update_rate >= 0.999
            && !self.nca_cfg.carry_fold_mode()
            && !self.nca_cfg.carry_launch_clamp;

        if can_use_fast_fused {
            let alpha = (self.nca_cfg.step_size * self.nca_cfg.damping_alpha) as f32;
            let leaky_lambda = self.nca_cfg.leaky_lambda.clamp(0.0, 0.99) as f32;
            let decay = 1.0 - leaky_lambda;
            let c_carry = self.nca_cfg.carry_channels.min(c);
            let c_hidden = c - c_carry;
            let mut perc_vec = vec![0.0f32; b * l * 3 * c];

            for _ in 0..steps {
                let (active, dst) = pp.split_mut();

                // 1. Perception stencil gathering: [x_i, x_{i-1}, x_i - x_{i-1}]
                for bi in 0..b {
                    let b_offset = bi * l * c;
                    let p_b_offset = bi * l * 3 * c;
                    for i in 0..l {
                        let cell_offset = b_offset + i * c;
                        let p_cell_offset = p_b_offset + i * 3 * c;

                        for ch in 0..c {
                            let curr = active[cell_offset + ch];
                            let left = if i > 0 {
                                active[b_offset + (i - 1) * c + ch]
                            } else {
                                0.0
                            };
                            let diff = curr - left;

                            perc_vec[p_cell_offset + ch] = curr;
                            perc_vec[p_cell_offset + c + ch] = left;
                            perc_vec[p_cell_offset + 2 * c + ch] = diff;
                        }
                    }
                }

                // 2. Linear MLP projections
                let perc_tensor = Tensor::from_slice(&perc_vec, (b, l, 3 * c), device)?;
                let h1 = self.dense1.forward(&perc_tensor)?;
                let h1_act = match self.nca_cfg.activation.as_str() {
                    "tanh" => h1.tanh()?,
                    _ => candle_nn::Activation::Gelu.forward(&h1)?,
                };
                let delta = self.dense_delta.forward(&h1_act)?.tanh()?;
                let gate = candle_nn::ops::sigmoid(&self.dense_gate.forward(&h1_act)?)?;
                let gated_delta = delta.mul(&gate)?;
                let delta_vec = gated_delta.flatten_all()?.to_vec1::<f32>()?;

                // 3. Fused carry advection + integration + quantization into inactive buffer

                for bi in 0..b {
                    let b_offset = bi * l * c;
                    for i in 0..l {
                        let cell_offset = b_offset + i * c;

                        // Stationary hidden channels: h_i * (1 - lambda)
                        for ch in 0..c_hidden {
                            let idx = cell_offset + ch;
                            let base_val = if leaky_lambda > 0.0 {
                                active[idx] * decay
                            } else {
                                active[idx]
                            };
                            dst[idx] = base_val + alpha * delta_vec[idx];
                        }

                        // Carry channels: hyperbolic advection shift
                        if c_carry > 0 {
                            let half = if self.nca_cfg.carry_bidirectional {
                                c_carry / 2
                            } else {
                                c_carry
                            };

                            for c_idx in 0..c_carry {
                                let ch = c_hidden + c_idx;
                                let idx = cell_offset + ch;

                                let base_val = if self.nca_cfg.carry_bidirectional {
                                    if c_idx < half {
                                        // Forward carry
                                        if self.nca_cfg.carry_skip_stride > 1 {
                                            let half_fwd = half / 2;
                                            if c_idx < half - half_fwd {
                                                if i >= 1 { active[b_offset + (i - 1) * c + ch] } else { 0.0 }
                                            } else {
                                                let k = self.nca_cfg.carry_skip_stride;
                                                if i >= k { active[b_offset + (i - k) * c + ch] } else { 0.0 }
                                            }
                                        } else {
                                            if i >= 1 { active[b_offset + (i - 1) * c + ch] } else { 0.0 }
                                        }
                                    } else {
                                        // Backward carry
                                        let bwd_len = c_carry - half;
                                        let bwd_idx = c_idx - half;
                                        if self.nca_cfg.carry_skip_stride > 1 {
                                            let half_bwd = bwd_len / 2;
                                            if bwd_idx < bwd_len - half_bwd {
                                                if i + 1 < l { active[b_offset + (i + 1) * c + ch] } else { 0.0 }
                                            } else {
                                                let k = self.nca_cfg.carry_skip_stride;
                                                if i + k < l { active[b_offset + (i + k) * c + ch] } else { 0.0 }
                                            }
                                        } else {
                                            if i + 1 < l { active[b_offset + (i + 1) * c + ch] } else { 0.0 }
                                        }
                                    }
                                } else if self.nca_cfg.carry_skip_stride > 1 {
                                    let half_fwd = c_carry / 2;
                                    if c_idx < c_carry - half_fwd {
                                        if i >= 1 { active[b_offset + (i - 1) * c + ch] } else { 0.0 }
                                    } else {
                                        let k = self.nca_cfg.carry_skip_stride;
                                        if i >= k { active[b_offset + (i - k) * c + ch] } else { 0.0 }
                                    }
                                } else {
                                    if i >= 1 { active[b_offset + (i - 1) * c + ch] } else { 0.0 }
                                };

                                let mut updated = base_val + alpha * delta_vec[idx];
                                match self.nca_cfg.carry_quantization.as_str() {
                                    "ste_sign" | "sign" | "ste_sign_ide" => {
                                        updated = if updated > 0.0 { 1.0 } else if updated < 0.0 { -1.0 } else { 0.0 };
                                    }
                                    "ste_round" | "round" | "ste_round_ide" => {
                                        updated = updated.round();
                                    }
                                    "bistable" => {
                                        updated = updated + 0.1 * updated * (1.0 - updated * updated);
                                    }
                                    _ => {}
                                }
                                dst[idx] = updated;
                            }
                        }
                    }
                }

                pp.toggle();
            }
        } else {
            // General fallback for non-causal / hierarchical configurations
            for _ in 0..steps {
                let morph = pp.to_morphogenic_field(device)?;
                let next_morph = self.step_field(&morph, device)?;
                let flat = next_morph.x.flatten_all()?.to_vec1::<f32>()?;
                pp.inactive_mut_slice().copy_from_slice(&flat);
                pp.toggle();
            }
        }

        Ok(())
    }

    /// Single step returning (new_field, delta_global, delta_local)
    /// where delta_global is the scale-invariant RMS displacement,
    /// and delta_local is the maximum normalized displacement of any individual cell.
    #[allow(dead_code)]
    pub fn step_with_observables(
        &self,
        field: &MorphogenicField,
        device: &Device,
    ) -> Result<(MorphogenicField, f32, f32)> {
        let (next_field, delta_global) = self.step_with_forcing(field, None, device)?;
        let total_displacement = (&next_field.x - &field.x)?;
        let cell_sq = total_displacement.sqr()?.mean(candle_core::D::Minus1)?; // [B, L]
        let delta_local = cell_sq.flatten_all()?.max(0)?.to_scalar::<f32>()?.sqrt();
        Ok((next_field, delta_global, delta_local))
    }

    /// Autonomous recurrent unroll with Dual-Metric Intrinsic Halting (Theorem 6).
    /// Halts when both delta_global < eps_global AND delta_local < eps_local
    /// for persistence_w consecutive steps, once step >= min_steps.
    /// Returns (final_field, total_steps_executed, converged).
    #[allow(dead_code)]
    pub fn develop_adaptive_dual(
        &self,
        initial: &MorphogenicField,
        min_steps: usize,
        max_steps: usize,
        persistence_w: usize,
        eps_global: f32,
        eps_local: f32,
        device: &Device,
    ) -> Result<(MorphogenicField, usize, bool)> {
        let mut current = initial.clone();
        let mut consecutive_converged = 0usize;
        let mut executed_steps = max_steps;
        let mut did_converge = false;

        for s in 1..=max_steps {
            let (next_field, delta_global, delta_local) = self.step_with_observables(&current, device)?;
            current = next_field;

            if s >= min_steps {
                if delta_global < eps_global && delta_local < eps_local {
                    consecutive_converged += 1;
                    if consecutive_converged >= persistence_w {
                        executed_steps = s;
                        did_converge = true;
                        break;
                    }
                } else {
                    consecutive_converged = 0;
                }
            }
        }

        Ok((current, executed_steps, did_converge))
    }


    /// Unrolls development with a sequence of external forcing tensors
    #[allow(dead_code)]
    pub fn develop_with_forcing(
        &self,
        initial: &MorphogenicField,
        forcings: &[Tensor],
        device: &Device,
    ) -> Result<MorphogenicField> {
        let mut current = initial.clone();
        for f in forcings {
            let (next, _) = self.step_with_forcing(&current, Some(f), device)?;
            current = next;
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_nca_step_and_viscosity() -> Result<()> {
        let dev = Device::Cpu;
        let mut nca_cfg = NcaConfig::default();
        let field_cfg = FieldConfig { seq_len: 16, channels: 8, periodic_boundary: true };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);

        // 1. Without viscosity
        let nca_inviscid = NeuralCellularAutomaton::new(vb.pp("nca"), &nca_cfg, &field_cfg)?;
        let initial = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let (next_field, norm) = nca_inviscid.step(&initial, &dev)?;
        assert_eq!(next_field.x.dims3()?, (1, 16, 8));
        assert!(norm >= 0.0);

        // 2. With Navier-Stokes viscosity
        nca_cfg.viscosity = 0.05;
        let nca_viscous = NeuralCellularAutomaton::new(vb.pp("nca"), &nca_cfg, &field_cfg)?;
        let (next_viscous, _) = nca_viscous.step(&initial, &dev)?;
        assert_eq!(next_viscous.x.dims3()?, (1, 16, 8));

        // 3. Forcing test
        let forcing_t = Tensor::zeros((1, 16, 8), candle_core::DType::F32, &dev)?;
        let (forced_field, _) = nca_viscous.step_with_forcing(&initial, Some(&forcing_t), &dev)?;
        assert_eq!(forced_field.x.dims3()?, (1, 16, 8));

        Ok(())
    }

    #[test]
    fn test_viscous_damping_attenuates_high_frequencies() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 4;
        let cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        // 1. Test k = L/4 mode where both central gradient and Laplacian are active
        let mut data_wave = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        let two_pi = 2.0 * std::f32::consts::PI;
        for i in 0..seq_len {
            let val = (two_pi * 4.0 * (i as f32) / (seq_len as f32)).sin();
            for c in 0..channels {
                data_wave[0][i][c] = val;
            }
        }
        let t_wave = Tensor::new(data_wave, &dev)?;
        let field_wave = MorphogenicField::from_tensor(t_wave, &cfg);

        let init_ens = field_wave.enstrophy()?;
        let init_palin = field_wave.palinstrophy()?;
        assert!(init_ens > 0.0);
        assert!(init_palin > 0.0);

        // Damping with moderate viscosity (effective diffusion D = 0.05 * 0.5 = 0.025)
        let damped_wave = NeuralCellularAutomaton::apply_viscous_dissipation(&field_wave.x, 0.025)?;
        let field_damped = MorphogenicField::from_tensor(damped_wave, &cfg);
        assert!(field_damped.enstrophy()? < init_ens, "Viscosity must attenuate enstrophy");
        assert!(field_damped.palinstrophy()? < init_palin, "Viscosity must attenuate palinstrophy");

        // 2. Test Nyquist mode (k = L/2: +1, -1, +1, -1) under high viscosity (D = 0.30 and D = 1.0)
        // In the prior attempt without adaptive substepping, D = 0.60 exploded to 1889.5!
        let mut data_nyquist = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        for i in 0..seq_len {
            let sign = if i % 2 == 0 { 1.0f32 } else { -1.0f32 };
            for c in 0..channels {
                data_nyquist[0][i][c] = sign;
            }
        }
        let t_nyq = Tensor::new(data_nyquist, &dev)?;
        let field_nyq = MorphogenicField::from_tensor(t_nyq, &cfg);
        let init_palin_nyq = field_nyq.palinstrophy()?;
        assert!(init_palin_nyq > 0.0);

        // High viscosity D = 0.30 (nu = 0.60, alpha = 0.5)
        let damped_high = NeuralCellularAutomaton::apply_viscous_dissipation(&field_nyq.x, 0.30)?;
        let field_high = MorphogenicField::from_tensor(damped_high, &cfg);
        let palin_high = field_high.palinstrophy()?;
        assert!(palin_high < init_palin_nyq, "Higher viscosity must damp palinstrophy");
        assert!(field_high.max_amplitude()? < field_nyq.max_amplitude()?);
        assert!(!palin_high.is_nan());

        // Extreme viscosity D = 1.00 (strongly extinguished by substepping)
        let damped_extreme = NeuralCellularAutomaton::apply_viscous_dissipation(&field_nyq.x, 1.00)?;
        let field_extreme = MorphogenicField::from_tensor(damped_extreme, &cfg);
        assert!(field_extreme.palinstrophy()? < palin_high);
        assert!(field_extreme.max_amplitude()? < 0.05, "Extreme viscosity must strongly extinguish Nyquist mode");

        Ok(())
    }

    #[test]
    fn test_macro_recursive_feedback_loop_lightcone_expansion() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: false };

        // 1. Test Baseline Local NCA (no feedback)
        let nca_cfg_local = NcaConfig {
            feedback_mode: "none".to_string(),
            ..NcaConfig::default()
        };
        let varmap_local = VarMap::new();
        let vb_local = VarBuilder::from_varmap(&varmap_local, candle_core::DType::F32, &dev);
        let nca_local = NeuralCellularAutomaton::new(vb_local, &nca_cfg_local, &field_cfg)?;

        let zeros_field = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let (step_local_zeros, _) = nca_local.step(&zeros_field, &dev)?;

        // Impulse at position 0 only: [1, 16, 8]
        let mut impulse_data = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        for c in 0..channels {
            impulse_data[0][0][c] = 5.0;
        }
        let impulse_field = MorphogenicField::from_tensor(Tensor::new(impulse_data, &dev)?, &field_cfg);

        // Run 1 step of local-only NCA
        let (step_local_impulse, _) = nca_local.step(&impulse_field, &dev)?;
        // Cell 8 is distance 8 away: with zero boundary and radius 1, cell 8's causal difference MUST be exactly 0.0!
        let diff_local = (&step_local_impulse.x.narrow(1, 8, 1)? - &step_local_zeros.x.narrow(1, 8, 1)?)?;
        let causal_effect_local = diff_local.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert_eq!(causal_effect_local, 0.0, "Local-only NCA cannot causally influence distant cell in 1 step");

        // 2. Test Macro Recursive Feedback NCA
        let nca_cfg_feedback = NcaConfig {
            feedback_mode: "global_pool".to_string(),
            ..NcaConfig::default()
        };
        let varmap_fb = VarMap::new();
        let vb_fb = VarBuilder::from_varmap(&varmap_fb, candle_core::DType::F32, &dev);
        let nca_fb = NeuralCellularAutomaton::new(vb_fb, &nca_cfg_feedback, &field_cfg)?;

        let (step_fb_zeros, _) = nca_fb.step(&zeros_field, &dev)?;
        let (step_fb_impulse, _) = nca_fb.step(&impulse_field, &dev)?;
        assert!(step_fb_impulse.slow_state.is_some(), "Feedback NCA must maintain macroscopic slow state");

        // In feedback NCA, position 0's impulse is pooled to macro state s_1, and perceived by all cells
        let diff_fb = (&step_fb_impulse.x.narrow(1, 8, 1)? - &step_fb_zeros.x.narrow(1, 8, 1)?)?;
        let causal_effect_fb = diff_fb.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert!(causal_effect_fb > 1e-6, "Feedback NCA enables instantaneous O(1) global causal communication");

        Ok(())
    }

    #[test]
    fn test_dual_timescale_leaky_contraction() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            feedback_mode: "dual_timescale".to_string(),
            feedback_weight: 0.2,
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let initial = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let developed = nca.develop(&initial, 8, &dev)?;
        assert!(developed.slow_state.is_some());
        let norm = developed.x.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        assert!(norm.is_finite());

        Ok(())
    }

    #[test]
    fn test_state_norm_compact_manifold() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            state_norm: "rms".to_string(),
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        // Seed with random large tensor
        let rand_tensor = Tensor::randn(0.0f32, 10.0f32, (1, seq_len, channels), &dev)?;
        let initial = MorphogenicField::from_tensor(rand_tensor, &field_cfg);

        // Unroll 64 autonomous steps
        let rolled = nca.develop(&initial, 64, &dev)?;

        // Under RMS normalization, each cell's mean square over channels must be ~1.0
        let cell_rms = rolled.x.sqr()?.mean_keepdim(candle_core::D::Minus1)?.to_vec3::<f32>()?;
        for b in 0..1 {
            for l in 0..seq_len {
                let rms_sq = cell_rms[b][l][0];
                assert!((rms_sq - 1.0).abs() < 1e-3, "Each cell vector must lie on the RMS unit sphere");
            }
        }

        // Energy must be exactly 0.5 * mean(sq) = 0.5 * 1.0 = 0.5
        let energy = rolled.energy()?;
        assert!((energy - 0.5).abs() < 1e-3, "Field energy on compact manifold must be bounded to 0.5");

        Ok(())
    }

    #[test]
    fn test_coordinate_channel_and_counterfactuals() -> Result<()> {
        let dev = Device::Cpu;
        let b = 2;
        let l = 5;

        // 1. Coordinate generation
        let intact = NeuralCellularAutomaton::generate_coordinates(b, l, None, &dev)?;
        let intact_v = intact.to_vec3::<f32>()?;
        assert_eq!(intact_v[0][0][0], -1.0);
        assert_eq!(intact_v[0][2][0], 0.0);
        assert_eq!(intact_v[0][4][0], 1.0);

        let zeroed = NeuralCellularAutomaton::generate_coordinates(b, l, Some("zeroed"), &dev)?;
        let zeroed_v = zeroed.to_vec3::<f32>()?;
        for cell in 0..l {
            assert_eq!(zeroed_v[0][cell][0], 0.0);
        }

        let constant = NeuralCellularAutomaton::generate_coordinates(b, l, Some("constant"), &dev)?;
        let constant_v = constant.to_vec3::<f32>()?;
        for cell in 0..l {
            assert_eq!(constant_v[0][cell][0], 0.5);
        }

        let reversed = NeuralCellularAutomaton::generate_coordinates(b, l, Some("reversed"), &dev)?;
        let reversed_v = reversed.to_vec3::<f32>()?;
        assert_eq!(reversed_v[0][0][0], 1.0);
        assert_eq!(reversed_v[0][4][0], -1.0);

        // 2. NCA Perception dimension
        let channels = 8;
        let field_cfg = FieldConfig { seq_len: l, channels, periodic_boundary: true };
        let nca_cfg = NcaConfig {
            coord_channel: true,
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let x = Tensor::zeros((b, l, channels), candle_core::DType::F32, &dev)?;
        let perc = nca.perceive(&x)?;
        assert_eq!(perc.dims3()?, (b, l, 3 * channels + 1));

        Ok(())
    }

    #[test]
    fn test_coord_train_mode_sham_constant() -> Result<()> {
        // Training-time sham: constant coordinate channel (same +1 dimension,
        // no positional information). Baseline (mode None) must remain intact.
        let dev = Device::Cpu;
        let l = 16;
        let channels = 4;
        let field_cfg = FieldConfig { seq_len: l, channels, periodic_boundary: true };

        // Sham: constant mode -> last perception channel is 0.5 everywhere.
        let nca_cfg_sham = NcaConfig {
            coord_channel: true,
            coord_train_mode: Some("constant".to_string()),
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca_sham = NeuralCellularAutomaton::new(vb, &nca_cfg_sham, &field_cfg)?;
        let x = Tensor::zeros((1, l, channels), candle_core::DType::F32, &dev)?;
        let perc_sham = nca_sham.perceive(&x)?;
        assert_eq!(perc_sham.dims3()?, (1, l, 3 * channels + 1));
        let sham_coords = perc_sham.narrow(2, 3 * channels, 1)?.to_vec3::<f32>()?;
        for i in 0..l {
            assert!((sham_coords[0][i][0] - 0.5).abs() < 1e-6, "constant sham at {i}");
        }

        // Baseline: mode None -> intact centered coordinates in [-1, 1].
        let nca_cfg_intact = NcaConfig {
            coord_channel: true,
            coord_train_mode: None,
            ..NcaConfig::default()
        };
        let varmap2 = VarMap::new();
        let vb2 = VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let nca_intact = NeuralCellularAutomaton::new(vb2, &nca_cfg_intact, &field_cfg)?;
        let perc_intact = nca_intact.perceive(&x)?;
        let intact_coords = perc_intact.narrow(2, 3 * channels, 1)?.to_vec3::<f32>()?;
        assert!((intact_coords[0][0][0] - (-1.0)).abs() < 1e-6, "first position = -1");
        assert!((intact_coords[0][l - 1][0] - 1.0).abs() < 1e-6, "last position = +1");
        for i in 0..l {
            let expected = 2.0 * (i as f32) / ((l - 1) as f32) - 1.0;
            assert!(
                (intact_coords[0][i][0] - expected).abs() < 1e-6,
                "coordinate formula at {i}"
            );
        }

        // L=1 edge case: single position coordinate must be defined (0.0).
        let field_cfg1 = FieldConfig { seq_len: 1, channels, periodic_boundary: true };
        let nca_cfg1 = NcaConfig { coord_channel: true, ..NcaConfig::default() };
        let varmap3 = VarMap::new();
        let vb3 = VarBuilder::from_varmap(&varmap3, candle_core::DType::F32, &dev);
        let nca1 = NeuralCellularAutomaton::new(vb3, &nca_cfg1, &field_cfg1)?;
        let x1 = Tensor::zeros((1, 1, channels), candle_core::DType::F32, &dev)?;
        let perc1 = nca1.perceive(&x1)?;
        assert_eq!(perc1.dims3()?, (1, 1, 3 * channels + 1));

        // Zeroed training mode: channel present but identically 0.
        let nca_cfg_zero = NcaConfig {
            coord_channel: true,
            coord_train_mode: Some("zeroed".to_string()),
            ..NcaConfig::default()
        };
        let varmap4 = VarMap::new();
        let vb4 = VarBuilder::from_varmap(&varmap4, candle_core::DType::F32, &dev);
        let nca_zero = NeuralCellularAutomaton::new(vb4, &nca_cfg_zero, &field_cfg)?;
        let perc_zero = nca_zero.perceive(&x)?;
        let zero_coords = perc_zero.narrow(2, 3 * channels, 1)?.to_vec3::<f32>()?;
        for i in 0..l {
            assert_eq!(zero_coords[0][i][0], 0.0, "zeroed sham at {i}");
        }

        Ok(())
    }

    #[test]
    fn test_hierarchy_zero_dc_response() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 2,
            macro_channels: 4,
            macro_period: 2,
            feedback_weight: 0.1,
            macro_coupling: "perception".to_string(),
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        // Construct a constant macro field: M_j = 3.5 for all j in 0..8
        let macro_len = 8;
        let const_macro = (Tensor::ones((1, macro_len, 4), candle_core::DType::F32, &dev)? * 3.5)?;
        let zero_micro = Tensor::zeros((1, seq_len, channels), candle_core::DType::F32, &dev)?;

        // Perceive with the constant macro state
        let perc = nca.perceive_with_feedback(&zero_micro, Some(&const_macro))?;
        // Micro perception has channels: 3 * C + C_M = 3 * 8 + 4 = 28
        assert_eq!(perc.dims3()?, (1, seq_len, 28));

        // The last 4 channels correspond to scaled_w.
        // Under a spatially constant macro field, the local difference stencil
        // w_j = M_j - 0.5 * (M_{j-1} + M_{j+1}) = 3.5 - 3.5 = 0.0!
        let w_channels = perc.narrow(2, 24, 4)?;
        let w_rms = w_channels.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        assert!(
            w_rms < 1e-6,
            "Local difference stencil must produce EXACT zero response to constant DC field, got RMS {}",
            w_rms
        );

        Ok(())
    }

    #[test]
    fn test_hierarchy_step_and_slow_clock() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 2,
            macro_channels: 4,
            macro_period: 2, // macro updates every 2 ticks
            step_size: 0.5,
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let initial_field = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        assert_eq!(initial_field.tick, 0);

        // Step 1: field.tick goes from 0 -> 1.
        // (0 + 1) % 2 == 1 != 0, so macro clock is not active; macro state remains all zeros
        let (field_step1, _) = nca.step(&initial_field, &dev)?;
        assert_eq!(field_step1.tick, 1);
        let m1 = field_step1.slow_state.as_ref().expect("macro state should exist");
        assert_eq!(m1.dims3()?, (1, 8, 4));
        let m1_norm = m1.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert_eq!(m1_norm, 0.0, "Macro state should remain frozen on non-macro ticks");

        // Step 2: field.tick goes from 1 -> 2.
        // (1 + 1) % 2 == 0, so macro clock fires! Macro field updates
        let (field_step2, _) = nca.step(&field_step1, &dev)?;
        assert_eq!(field_step2.tick, 2);
        assert!(field_step2.slow_state.is_some());

        Ok(())
    }

    #[test]
    fn test_hierarchy_degenerate_stride_control() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        // Degenerate control: stride s = 1 (no spatial coarsening)
        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 1,
            macro_channels: 4,
            macro_period: 2,
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let initial_field = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let (field_step1, _) = nca.step(&initial_field, &dev)?;
        let m1 = field_step1.slow_state.as_ref().expect("macro state should exist");
        // With stride 1, macro grid length is L_M = L = 16
        assert_eq!(m1.dims3()?, (1, 16, 4));

        Ok(())
    }

    #[test]
    fn test_hierarchy_parameter_breakdown() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 2,
            macro_channels: 4,
            macro_coupling: "state_derivative".to_string(),
            hidden_dim: 32,
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let breakdown = nca.parameter_breakdown();
        let names: Vec<&str> = breakdown.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"macro_hierarchy_mlp_dense1"));
        assert!(names.contains(&"macro_hierarchy_directional_delta"));
        assert!(names.contains(&"macro_state_derivative_inject"));
        assert!(names.contains(&"macro_local_highpass_modulation"));

        Ok(())
    }

    #[test]
    fn test_hierarchy_walsh_and_state_derivative() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        // Test with Walsh downsampling and state-derivative coupling
        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 2,
            macro_channels: 4,
            macro_downsampler: "walsh".to_string(),
            macro_coupling: "state_derivative".to_string(),
            macro_gamma: 0.2,
            macro_lambda: 0.1,
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let mut field = MorphogenicField::zeros(2, &field_cfg, &dev)?;
        // Run 4 steps (two macro update cycles)
        for _ in 0..4 {
            let (next_f, _) = nca.step(&field, &dev)?;
            field = next_f;
        }

        // Verify state is bounded in [-1, 1] (barrier function)
        let max_abs = field.x.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(max_abs <= 1.05, "State must remain bounded within barrier, got {}", max_abs);

        // Verify macro state exists and has correct dimensions [B, L/2, C_M]
        let m = field.slow_state.expect("Macro state should exist");
        assert_eq!(m.dims3()?, (2, 8, 4));

        Ok(())
    }

    #[test]
    fn test_causal_stencil_dag_property() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: false };

        let nca_cfg = NcaConfig {
            causal_stencil: true,
            viscosity: 0.05, // Upwind dissipation must also be strictly causal
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        // Base field with random initial values
        let x1 = Tensor::randn(0.0f32, 1.0f32, (1, seq_len, channels), &dev)?;
        let f1 = MorphogenicField::from_tensor(x1, &field_cfg);

        // Perturbed field: identical to f1 everywhere except at slot k=8
        let mut x_pert_vals = f1.x.to_vec3::<f32>()?;
        for c in 0..channels {
            x_pert_vals[0][8][c] += 1.0;
        }
        let f2 = MorphogenicField::from_tensor(
            Tensor::from_vec(
                x_pert_vals.into_iter().flatten().flatten().collect(),
                (1, seq_len, channels),
                &dev,
            )?,
            &field_cfg,
        );

        // Step both fields forward by multiple developmental ticks
        let mut curr1 = f1;
        let mut curr2 = f2;
        for _ in 0..5 {
            let (next1, _) = nca.step(&curr1, &dev)?;
            let (next2, _) = nca.step(&curr2, &dev)?;
            curr1 = next1;
            curr2 = next2;
        }

        // Check difference: For all cells i < 8, diff must be strictly 0.0 (exact DAG property)
        let diff = (&curr1.x - &curr2.x)?.abs()?;
        let diff_vals = diff.to_vec3::<f32>()?;

        for i in 0..8 {
            for c in 0..channels {
                let d = diff_vals[0][i][c];
                assert_eq!(
                    d, 0.0,
                    "Cell {} (upstream of perturbation at 8) was affected by perturbation (diff = {})! Anti-causal leakage detected!",
                    i, d
                );
            }
        }

        // Cell 8 and downstream cells MUST reflect the perturbation
        let mut downstream_diff: f32 = 0.0;
        for i in 8..seq_len {
            for c in 0..channels {
                downstream_diff += diff_vals[0][i][c];
            }
        }
        assert!(
            downstream_diff > 1e-4,
            "Perturbation failed to propagate forward downstream (downstream diff = {})",
            downstream_diff
        );

        Ok(())
    }

    #[test]
    fn test_quantize_carry_ops() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::from_slice(&[-1.4f32, -0.6, 0.0, 0.2, 0.7, 1.4], (1, 6, 1), &dev)?;

        // 1. None
        let q_none = NeuralCellularAutomaton::quantize_carry(&t, "none")?;
        assert_eq!(q_none.to_vec3::<f32>()?, t.to_vec3::<f32>()?);

        // 2. STE Round
        let q_round = NeuralCellularAutomaton::quantize_carry(&t, "ste_round")?;
        let round_vals = q_round.to_vec3::<f32>()?;
        assert_eq!(round_vals, vec![vec![vec![-1.0], vec![-1.0], vec![0.0], vec![0.0], vec![1.0], vec![1.0]]]);

        // 3. STE Sign
        let q_sign = NeuralCellularAutomaton::quantize_carry(&t, "ste_sign")?;
        let sign_vals = q_sign.to_vec3::<f32>()?;
        assert_eq!(sign_vals, vec![vec![vec![-1.0], vec![-1.0], vec![0.0], vec![1.0], vec![1.0], vec![1.0]]]);

        // 4. Bistable
        let q_bistable = NeuralCellularAutomaton::quantize_carry(&t, "bistable")?;
        let bistable_vals = q_bistable.to_vec3::<f32>()?;
        // For 0.7: 0.7 + 0.1 * 0.7 * (1 - 0.49) = 0.7 + 0.0357 = 0.7357 (driven closer to 1.0)
        assert!((bistable_vals[0][4][0] - 0.7357).abs() < 1e-3);
        // For 1.4: 1.4 + 0.1 * 1.4 * (1 - 1.96) = 1.4 - 0.1344 = 1.2656 (restored toward 1.0)
        assert!((bistable_vals[0][5][0] - 1.2656).abs() < 1e-3);

        Ok(())
    }

    /// Record the historical gradient semantics without silently changing old
    /// training/checkpoint behavior. These modes are NOT identity-gradient STEs.
    #[test]
    fn legacy_quantization_backward_is_zero_not_identity_ste() -> Result<()> {
        for mode in ["ste_sign", "ste_round"] {
            let x = candle_core::Var::from_slice(&[-0.7f32, 0.3, 1.2], 3, &Device::Cpu)?;
            let loss = NeuralCellularAutomaton::quantize_carry(&x, mode)?.sum_all()?;
            let grads = loss.backward()?;
            assert_eq!(grads.get(&x).expect("gradient path").to_vec1::<f32>()?, vec![0.0; 3]);
        }
        Ok(())
    }

    #[test]
    fn test_develop_adaptive_dual_halting() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);

        let nca_cfg = NcaConfig {
            hidden_dim: 32,
            step_size: 0.5,
            update_rate: 1.0,
            activation: "gelu".to_string(),
            viscosity: 0.05,
            leaky_lambda: 0.1, // Strong contraction ensures settling
            causal_stencil: true,
            carry_channels: 16,
            carry_skip_stride: 4,
            carry_bidirectional: true,
            carry_quantization: "none".to_string(),
            ..NcaConfig::default()
        };
        let field_cfg = FieldConfig {
            seq_len: 16,
            channels: 32,
            periodic_boundary: false,
        };

        let nca = NeuralCellularAutomaton::new(vs, &nca_cfg, &field_cfg)?;
        let initial = MorphogenicField::zeros(1, &field_cfg, &dev)?;

        // Run adaptive halting with W=3, eps_global=0.05, eps_local=0.1
        let (_field, steps, did_converge) = nca.develop_adaptive_dual(
            &initial,
            4,   // min_steps
            24,  // max_steps
            3,   // persistence_w
            0.05, // eps_global
            0.10, // eps_local
            &dev,
        )?;

        assert!(steps >= 4, "Must observe min_steps warmup");
        assert!(steps <= 24, "Cannot exceed max_steps");
        assert!(did_converge, "Field should converge under leaky contraction");

        Ok(())
    }

    #[test]
    fn test_ping_pong_exact_numerical_equivalence() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);

        let nca_cfg = NcaConfig {
            hidden_dim: 96,
            step_size: 0.5,
            update_rate: 1.0,
            activation: "gelu".to_string(),
            viscosity: 0.0,
            leaky_lambda: 0.0,
            causal_stencil: true,
            carry_channels: 32,
            carry_skip_stride: 4,
            carry_bidirectional: true,
            carry_quantization: "ste_sign".to_string(),
            ..NcaConfig::default()
        };
        let field_cfg = FieldConfig {
            seq_len: 16,
            channels: 64,
            periodic_boundary: false,
        };

        let nca = NeuralCellularAutomaton::new(vs, &nca_cfg, &field_cfg)?;

        // Create initial random field [1, 16, 64]
        let initial_data = Tensor::randn(0.0f32, 1.0f32, (1, 16, 64), &dev)?;
        let initial_field = MorphogenicField::from_tensor(initial_data, &field_cfg);

        // Run naive develop for 8 steps
        let naive_out = nca.develop(&initial_field, 8, &dev)?;

        // Run ping-pong develop for 8 steps
        let mut pp = PingPongField::from_morphogenic_field(&initial_field)?;
        nca.develop_ping_pong(&mut pp, 8, &dev)?;
        let pp_out = pp.to_morphogenic_field(&dev)?;

        // Assert exact numerical equivalence
        let diff = (&naive_out.x - &pp_out.x)?.abs()?;
        let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;

        assert!(
            max_diff < 1e-5,
            "PingPongField developed output diverged from naive develop: max_diff = {}",
            max_diff
        );

        Ok(())
    }

    // ==================================================================
    // Relay-Fold Transport (RFT) battery
    // ==================================================================

    fn zero_all_vars(varmap: &VarMap) -> Result<()> {
        for var in varmap.all_vars() {
            let zeros = Tensor::zeros(var.dims().to_vec(), var.dtype(), var.device())?;
            var.set(&zeros)?;
        }
        Ok(())
    }

    fn set_var(varmap: &VarMap, name: &str, value: &Tensor) -> Result<()> {
        let data = varmap.data().lock().unwrap();
        let var = data
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("variable '{name}' not found in VarMap"))?;
        var.set(value)?;
        Ok(())
    }

    /// Identity-gradient STE modes must match the legacy forward projection but
    /// propagate dy/dcarry == 1 (the legacy modes keep dy/dcarry == 0).
    #[test]
    fn test_identity_ste_modes_have_unit_gradient() -> Result<()> {
        let dev = Device::Cpu;
        for (mode, legacy) in [("ste_round_ide", "ste_round"), ("ste_sign_ide", "ste_sign")] {
            let x = candle_core::Var::from_slice(&[-0.7f32, 0.3, 1.2], 3, &dev)?;
            let q = NeuralCellularAutomaton::quantize_carry(&x, mode)?;
            let q_vals = q.to_vec1::<f32>()?;
            let legacy_vals =
                NeuralCellularAutomaton::quantize_carry(&x, legacy)?.to_vec1::<f32>()?;
            assert_eq!(q_vals, legacy_vals, "{mode} forward must match {legacy}");
            let loss = q.sum_all()?;
            let grads = loss.backward()?;
            let g = grads.get(&x).expect("gradient path").to_vec1::<f32>()?;
            assert_eq!(
                g,
                vec![1.0f32; 3],
                "{mode} must have identity STE gradient (dy/dcarry == 1)"
            );
        }
        Ok(())
    }

    /// `carry_quantization = "fold"` (RFT) applies speed k UNIFORMLY to every
    /// carry channel (front advances exactly k cells/tick); legacy k>1 modes
    /// keep the mixed slow/fast half-split so k is not a uniform speed there.
    #[test]
    fn test_uniform_carry_stride_speed_knob() -> Result<()> {
        let dev = Device::Cpu;
        let l = 16usize;
        let channels = 24usize; // 8 hidden + 16 carry
        let hidden = 8usize;
        let cc = 16usize;
        let field_cfg = FieldConfig {
            seq_len: l,
            channels,
            periodic_boundary: false,
        };

        // (a) unidirectional RFT: every carry channel's transported front
        //     advances exactly k cells per tick.
        for k in [1usize, 2, 4] {
            let cfg = NcaConfig {
                hidden_dim: 8,
                causal_stencil: true,
                carry_channels: cc,
                carry_skip_stride: k,
                carry_quantization: "fold".to_string(),
                ..NcaConfig::default()
            };
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
            let nca = NeuralCellularAutomaton::new(vb, &cfg, &field_cfg)?;
            zero_all_vars(&varmap)?;

            let mut data = vec![0f32; l * channels];
            for ch in hidden..channels {
                data[ch] = 1.0; // single-hot seed, cell 0 only
            }
            let initial = MorphogenicField::from_tensor(
                Tensor::from_vec(data, (1, l, channels), &dev)?,
                &field_cfg,
            );

            for t in 1..=3usize {
                let rolled = nca.develop(&initial, t, &dev)?;
                let vals = rolled.x.to_vec3::<f32>()?;
                let front = t * k;
                for ch in hidden..channels {
                    let mut farthest = None;
                    for i in 0..l {
                        if vals[0][i][ch].abs() > 1e-6 {
                            farthest = Some(i);
                        }
                    }
                    assert_eq!(
                        farthest,
                        Some(front),
                        "uniform k={k}, tick {t}: carry channel {ch} must reach exactly cell {front}"
                    );
                }
            }

            // Acceptance readback: after 1 tick the seed value is AT cell k
            // (transported, gate=1/2 fixed by the zeroed fold gate) and has not
            // passed beyond it.
            let rolled1 = nca.develop(&initial, 1, &dev)?;
            let v1 = rolled1.x.to_vec3::<f32>()?;
            for ch in hidden..channels {
                assert!(
                    (v1[0][k][ch] - 0.5).abs() < 1e-6,
                    "k={k}: transported value must sit on cell {k} after 1 tick"
                );
                if k + 1 < l {
                    assert_eq!(v1[0][k + 1][ch], 0.0, "k={k}: nothing beyond cell {k} after 1 tick");
                }
            }
        }

        // (b) bidirectional RFT: both directions advance at the same speed k.
        {
            let k = 2usize;
            let cfg = NcaConfig {
                hidden_dim: 8,
                causal_stencil: true,
                carry_channels: cc,
                carry_skip_stride: k,
                carry_bidirectional: true,
                carry_quantization: "fold".to_string(),
                ..NcaConfig::default()
            };
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
            let nca = NeuralCellularAutomaton::new(vb, &cfg, &field_cfg)?;
            zero_all_vars(&varmap)?;

            let mut data = vec![0f32; l * channels];
            for ch in hidden..channels {
                data[ch] = 1.0; // cell 0 seed
                data[(l - 1) * channels + ch] = 1.0; // cell 15 seed
            }
            let initial = MorphogenicField::from_tensor(
                Tensor::from_vec(data, (1, l, channels), &dev)?,
                &field_cfg,
            );

            for t in 1..=2usize {
                let rolled = nca.develop(&initial, t, &dev)?;
                let vals = rolled.x.to_vec3::<f32>()?;
                for ch in hidden..hidden + cc / 2 {
                    let mut farthest = 0usize;
                    for i in 0..l {
                        if vals[0][i][ch].abs() > 1e-6 {
                            farthest = i;
                        }
                    }
                    assert_eq!(farthest, t * k, "fwd channel {ch} front at tick {t}");
                }
                for ch in hidden + cc / 2..channels {
                    let mut nearest = l - 1;
                    for i in 0..l {
                        if vals[0][i][ch].abs() > 1e-6 {
                            nearest = i;
                            break;
                        }
                    }
                    assert_eq!(nearest, l - 1 - t * k, "bwd channel {ch} front at tick {t}");
                }
            }
        }

        // (c) Contrast: legacy (non-fold) k>1 keeps the historical slow/fast
        //     half-channel split — k is a MIXTURE, not a uniform speed knob.
        {
            let cfg = NcaConfig {
                hidden_dim: 8,
                causal_stencil: true,
                carry_channels: cc,
                carry_skip_stride: 2,
                carry_quantization: "none".to_string(),
                ..NcaConfig::default()
            };
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
            let nca = NeuralCellularAutomaton::new(vb, &cfg, &field_cfg)?;
            zero_all_vars(&varmap)?;
            let mut data = vec![0f32; l * channels];
            for ch in hidden..channels {
                data[ch] = 1.0;
            }
            let initial = MorphogenicField::from_tensor(
                Tensor::from_vec(data, (1, l, channels), &dev)?,
                &field_cfg,
            );
            let rolled = nca.develop(&initial, 1, &dev)?;
            let vals = rolled.x.to_vec3::<f32>()?;
            let farthest = |ch: usize| -> usize {
                let mut f = 0usize;
                for i in 0..l {
                    if vals[0][i][ch].abs() > 1e-6 {
                        f = i;
                    }
                }
                f
            };
            assert_eq!(farthest(hidden), 1, "legacy slow half hops 1 cell/tick");
            assert_eq!(farthest(hidden + cc - 1), 2, "legacy fast half hops k cells/tick");
            assert_ne!(
                farthest(hidden),
                farthest(hidden + cc - 1),
                "legacy k>1 is a mixed slow/fast speed, not a uniform knob"
            );
        }

        Ok(())
    }

    /// Relay-Fold Transport accumulates: every relay cell folds its own known
    /// bit into the incoming carry. Engineered readout makes the far-cell value
    /// a closed-form weighted fold of ALL bits on the path, while pure additive
    /// (advective) transport copies the seed and leaves interior bits invisible.
    #[test]
    fn test_relay_fold_accumulates_known_bits_exactly() -> Result<()> {
        let dev = Device::Cpu;
        let l = 16usize;
        let channels = 8usize; // 7 hidden + 1 carry
        let hidden = 8usize; // hidden_dim
        let t_max = 8usize;
        let field_cfg = FieldConfig {
            seq_len: l,
            channels,
            periodic_boundary: false,
        };

        // Engineering: dense1 row 0 computes pre = 2*bit - 1 from the identity
        // read of state channel 0 (activation = tanh => h1_act[0] = s*tanh(1),
        // s = sign of the bit); the fold candidate reads h1_act[0] with unit
        // weight; the fold gate stays all-zero => g = sigmoid(0) = 1/2 exactly.
        // The relay rule therefore is exactly c_i = 1/2 c_{i-1} + 1/2 tanh(tanh(s_i)).
        let fold_cfg = NcaConfig {
            hidden_dim: hidden,
            activation: "tanh".to_string(),
            causal_stencil: true,
            carry_channels: 1,
            carry_quantization: "fold".to_string(),
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &fold_cfg, &field_cfg)?;
        zero_all_vars(&varmap)?;

        let mut w1 = vec![0f32; hidden * 3 * channels];
        w1[0] = 2.0; // row 0, col 0 = identity channel of bit channel
        set_var(
            &varmap,
            "dense1.weight",
            &Tensor::from_vec(w1, (hidden, 3 * channels), &dev)?,
        )?;
        let mut b1 = vec![0f32; hidden];
        b1[0] = -1.0;
        set_var(&varmap, "dense1.bias", &Tensor::from_vec(b1, hidden, &dev)?)?;
        let fold_in = 1 + hidden;
        let mut wc = vec![0f32; fold_in];
        wc[1] = 1.0; // [incoming(1) ; h1_act(hidden)] -> h1_act[0]
        set_var(
            &varmap,
            "carry_fold_cand.weight",
            &Tensor::from_vec(wc, (1, fold_in), &dev)?,
        )?;

        let make_initial = |bits: &[f32]| -> Result<MorphogenicField> {
            let mut data = vec![0f32; l * channels];
            for (idx, &bit) in bits.iter().enumerate() {
                data[(idx + 1) * channels] = bit; // bits live at cells 1..=len
            }
            data[channels - 1] = 0.5; // cell 0, carry channel: c0 = 1/2
            Ok(MorphogenicField::from_tensor(
                Tensor::from_vec(data, (1, l, channels), &dev)?,
                &field_cfg,
            ))
        };
        let read_carry = |f: &MorphogenicField, cell: usize| -> Result<f32> {
            Ok(f.x
                .narrow(1, cell, 1)?
                .narrow(2, channels - 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?[0])
        };

        let all_ones = vec![1.0f32; t_max];
        let kappa = (1.0f64.tanh()).tanh(); // |tanh(tanh(s))| for s in {-1, +1}
        let expected = |t: usize, bits: &[f32]| -> f64 {
            let mut v = 0.5f64.powi(t as i32) * 0.5; // 0.5^t * c0
            for m in 1..=t {
                let s = if bits[m - 1] > 0.5 { 1.0 } else { -1.0 };
                v += 0.5f64.powi((t - m + 1) as i32) * kappa * s;
            }
            v
        };

        // (a) Diagonal: after t ticks the t-th cell holds the folded prefix.
        for t in 1..=t_max {
            let rolled = nca.develop(&make_initial(&all_ones)?, t, &dev)?;
            let got = read_carry(&rolled, t)? as f64;
            let want = expected(t, &all_ones);
            assert!(
                (got - want).abs() < 1e-4,
                "fold prefix at tick {t}: got {got}, expected {want}"
            );
        }

        // (b) Flipping ONE interior bit changes the far cell by the exact
        //     predicted coefficient 2*kappa*0.5^(T-j+1) — the far cell reflects
        //     the folded sum of every known bit on the path.
        let rolled_ones = nca.develop(&make_initial(&all_ones)?, t_max, &dev)?;
        let got_ones = read_carry(&rolled_ones, t_max)? as f64;
        let mut flipped = all_ones.clone();
        flipped[3] = 0.0; // cell 4 bit flips (s_4: +1 -> -1)
        let rolled_flip = nca.develop(&make_initial(&flipped)?, t_max, &dev)?;
        let got_flip = read_carry(&rolled_flip, t_max)? as f64;
        assert!(
            (got_flip - expected(t_max, &flipped)).abs() < 1e-4,
            "flipped-bit fold: got {got_flip}, expected {}",
            expected(t_max, &flipped)
        );
        let predicted_delta = -2.0 * kappa * 0.5f64.powi(5); // 0.5^(8-4+1)
        assert!(
            ((got_flip - got_ones) - predicted_delta).abs() < 1e-4,
            "flip at cell 4 must shift the far carry by {predicted_delta}, got {}",
            got_flip - got_ones
        );

        // (c) Pure additive (advective) transport with the same engineered
        //     readout does NOT absorb interior bits: the far cell is a copy of
        //     the seed and the flip leaves NO trace.
        let add_cfg = NcaConfig {
            hidden_dim: hidden,
            activation: "tanh".to_string(),
            causal_stencil: true,
            carry_channels: 1,
            carry_quantization: "none".to_string(),
            ..NcaConfig::default()
        };
        let varmap2 = VarMap::new();
        let vb2 = VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let nca_add = NeuralCellularAutomaton::new(vb2, &add_cfg, &field_cfg)?;
        zero_all_vars(&varmap2)?;
        let mut w1b = vec![0f32; hidden * 3 * channels];
        w1b[0] = 2.0;
        set_var(
            &varmap2,
            "dense1.weight",
            &Tensor::from_vec(w1b, (hidden, 3 * channels), &dev)?,
        )?;
        let mut b1b = vec![0f32; hidden];
        b1b[0] = -1.0;
        set_var(&varmap2, "dense1.bias", &Tensor::from_vec(b1b, hidden, &dev)?)?;

        let rolled_add = nca_add.develop(&make_initial(&all_ones)?, t_max, &dev)?;
        let got_add = read_carry(&rolled_add, t_max)?;
        assert!(
            (got_add - 0.5).abs() < 1e-6,
            "pure advection only copies the seed, got {got_add}"
        );
        let rolled_add_flip = nca_add.develop(&make_initial(&flipped)?, t_max, &dev)?;
        let got_add_flip = read_carry(&rolled_add_flip, t_max)?;
        assert!(
            (got_add_flip - got_add).abs() < 1e-6,
            "pure advection leaves interior bits invisible (delta {})",
            got_add_flip - got_add
        );

        Ok(())
    }

    /// RFT fold projections must be trainable: gradients flow back through the
    /// relay write (unlike the historical zero-gradient STE carry path).
    #[test]
    fn test_relay_fold_backward_gradients_flow() -> Result<()> {
        let dev = Device::Cpu;
        let channels = 24usize;
        let cc = 16usize;
        let field_cfg = FieldConfig {
            seq_len: 16,
            channels,
            periodic_boundary: false,
        };
        let cfg = NcaConfig {
            hidden_dim: 16,
            causal_stencil: true,
            carry_channels: cc,
            carry_skip_stride: 2,
            carry_bidirectional: true,
            carry_quantization: "fold".to_string(),
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &cfg, &field_cfg)?;
        let initial = MorphogenicField::from_tensor(
            Tensor::randn(0.0f32, 1.0f32, (1, 16, channels), &dev)?,
            &field_cfg,
        );
        let out = nca.develop(&initial, 4, &dev)?;
        let loss = out.x.narrow(2, channels - cc, cc)?.sum_all()?;
        let grads = loss.backward()?;
        let data = varmap.data().lock().unwrap();
        for name in [
            "carry_fold_gate.weight",
            "carry_fold_gate.bias",
            "carry_fold_cand.weight",
            "carry_fold_cand.bias",
        ] {
            let var = data.get(name).expect("fold projection var exists");
            let g = grads.get(var).expect("gradient path to fold projection");
            let sum_abs = g.abs()?.sum_all()?.to_scalar::<f32>()?;
            assert!(
                sum_abs > 1e-8,
                "RFT fold projection '{name}' must receive gradient, got sum|g| = {sum_abs}"
            );
        }
        Ok(())
    }

    /// Launch dead-zone: legacy zero-pad only reaches q ≡ 0 (mod k); the
    /// clamped launch (opt-in field, forced by RFT) reaches every q in
    /// ceil(q/k) ticks.
    #[test]
    fn test_launch_dead_zone_and_clamped_launch() -> Result<()> {
        let dev = Device::Cpu;

        // (a) helper semantics on a small probe tensor.
        let probe_cfg = NcaConfig::default();
        let probe_field = FieldConfig {
            seq_len: 6,
            channels: 1,
            periodic_boundary: false,
        };
        let varmap0 = VarMap::new();
        let vb0 = VarBuilder::from_varmap(&varmap0, candle_core::DType::F32, &dev);
        let nca0 = NeuralCellularAutomaton::new(vb0, &probe_cfg, &probe_field)?;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 6, 1), &dev)?;
        assert_eq!(
            nca0.k_left_neighbor(&x, 2)?.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 0., 1., 2., 3., 4.]
        );
        assert_eq!(
            nca0.k_left_neighbor_clamped(&x, 2)?.flatten_all()?.to_vec1::<f32>()?,
            vec![1., 1., 1., 2., 3., 4.]
        );
        assert_eq!(
            nca0.k_right_neighbor(&x, 2)?.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 4., 5., 6., 0., 0.]
        );
        assert_eq!(
            nca0.k_right_neighbor_clamped(&x, 2)?.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 4., 5., 6., 6., 6.]
        );

        // (b) integration on the fast-half k=4 carry channels of the legacy
        //     (non-fold) transport, contrasting zero-pad vs clamped launch.
        let l = 16usize;
        let channels = 12usize;
        let hidden = 8usize;
        let cc = 4usize;
        let field_cfg = FieldConfig {
            seq_len: l,
            channels,
            periodic_boundary: false,
        };
        for clamp in [false, true] {
            let cfg = NcaConfig {
                hidden_dim: 8,
                causal_stencil: true,
                carry_channels: cc,
                carry_skip_stride: 4,
                carry_launch_clamp: clamp,
                ..NcaConfig::default()
            };
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
            let nca = NeuralCellularAutomaton::new(vb, &cfg, &field_cfg)?;
            zero_all_vars(&varmap)?;
            let mut data = vec![0f32; l * channels];
            for ch in hidden..channels {
                data[ch] = 1.0; // seed cell 0 only
            }
            let initial = MorphogenicField::from_tensor(
                Tensor::from_vec(data, (1, l, channels), &dev)?,
                &field_cfg,
            );

            let fast = hidden + 2; // carry-local channels 2,3 hop k cells (fast half)
            let t2 = nca.develop(&initial, 2, &dev)?;
            let v2 = t2.x.to_vec3::<f32>()?;
            if clamp {
                // 7 = 3 + 4 is off-lattice (7 mod 4 != 0): reachable at ceil(7/4)=2
                assert!(
                    (v2[0][7][fast] - 1.0).abs() < 1e-6,
                    "clamped launch must relay into cell 7 at tick 2"
                );
            } else {
                // dead zone: cell 7's k-hop lineage roots at cell 3 < k, which reads zero
                assert_eq!(
                    v2[0][7][fast], 0.0,
                    "legacy zero-pad cannot launch to off-lattice cell 7"
                );
                assert!(
                    (v2[0][8][fast] - 1.0).abs() < 1e-6,
                    "aligned cell 8 = 2k is still reached at tick 2"
                );
            }
            let t4 = nca.develop(&initial, 4, &dev)?;
            let v4 = t4.x.to_vec3::<f32>()?;
            if clamp {
                assert!(
                    (v4[0][15][fast] - 1.0).abs() < 1e-6,
                    "clamped launch reaches q=15 in ceil(15/4)=4 ticks"
                );
            } else {
                assert_eq!(
                    v4[0][15][fast], 0.0,
                    "legacy zero-pad never reaches q=15 (15 mod 4 != 0)"
                );
            }
        }

        Ok(())
    }

    /// The fused ping-pong row-walk must stay numerically equivalent to the
    /// general path for the new identity-gradient STE modes.
    #[test]
    fn test_ping_pong_equivalence_with_identity_ste() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);

        let nca_cfg = NcaConfig {
            hidden_dim: 96,
            step_size: 0.5,
            update_rate: 1.0,
            activation: "gelu".to_string(),
            viscosity: 0.0,
            leaky_lambda: 0.0,
            causal_stencil: true,
            carry_channels: 32,
            carry_skip_stride: 4,
            carry_bidirectional: true,
            carry_quantization: "ste_sign_ide".to_string(),
            ..NcaConfig::default()
        };
        let field_cfg = FieldConfig {
            seq_len: 16,
            channels: 64,
            periodic_boundary: false,
        };
        let nca = NeuralCellularAutomaton::new(vs, &nca_cfg, &field_cfg)?;

        let initial_data = Tensor::randn(0.0f32, 1.0f32, (1, 16, 64), &dev)?;
        let initial_field = MorphogenicField::from_tensor(initial_data, &field_cfg);

        let naive_out = nca.develop(&initial_field, 8, &dev)?;
        let mut pp = PingPongField::from_morphogenic_field(&initial_field)?;
        nca.develop_ping_pong(&mut pp, 8, &dev)?;
        let pp_out = pp.to_morphogenic_field(&dev)?;

        let max_diff = (&naive_out.x - &pp_out.x)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            max_diff < 1e-5,
            "ste_sign_ide ping-pong must match develop: max_diff = {max_diff}"
        );

        Ok(())
    }
}


