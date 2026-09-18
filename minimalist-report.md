# MINIMALISM AUDIT & OCCAM'S RAZOR EXECUTION REPORT
**Target Repository**: `/data/data/com.termux/files/home/projects/titan_text`  
**Auditor**: Minimalist Executioner  
**Status**: Comprehensive Codebase & Theory Audit  
**Date**: September 14, 2026  

---

## Executive Summary: The Causal Razor

Titan Text is a 1D recurrent sequence model parameterized as a weight-tied discrete cellular automaton. It unrolls across a 1D spatial lattice using local convolution stencils (perception) and a 2-layer gated MLP update function.

That is the **entire causal mechanism** of the system.

Surrounding this simple mechanism is an elaborate theoretical superstructure:
1. **Pseudo-Physics Baggage**: Millennium Prize Navier-Stokes singularity claims, Beale-Kato-Majda blow-up criteria, enstrophy/palinstrophy metrics, and Kolmogorov energy cascade calculations applied to a **1D discrete 32-cell lattice** where curl is identically zero, vortex stretching is mathematically impossible, and updates are strictly bounded Lipschitz steps ($\|\Delta \mathbf{x}\|_\infty \le 0.5$).
2. **Dead Architectural Decoration**: Mixed-timescale slow/fast channel gating (`slow_channel_fraction`, `slow_cadence`) that is never used during training, defaults to 0.0, and exists solely as dead code in `latent.rs`.
3. **Ghost Configuration & Dead CLI Flags**: Config fields that are never read (`periodic_boundary`, `model`, `observation_interval`), 12 CLI lesion flags accepted by `rollout` but ignored in code, and duplicate commands (`falsify` vs `probe`).
4. **Untrained Baseline Comparisons**: Benchmark commands comparing trained NCA models against randomly initialized, untrained Transformers and GRUs with zero training iterations.
5. **Inflated Dynamical Narrative**: Pointwise activation curvature ($f''(x)$) branded as "Nonlinear Upward Coupling ($Q$ Norm)"; distinct forward passes of different prompts branded as "Attractor Basins and Hysteresis"; and softmax temperature collapse under norm inflation branded as "Dynamic Semantic Association Emergence."

Every audited component is classified below according to the three required categories:
- **`SAFE TO DELETE`**: Zero impact on sequence modeling performance; eliminates confusion, dead code, and runtime bloat.
- **`CANDIDATE FOR SIMPLIFICATION`**: Can be replaced by a simple 5-line primitive or needs wiring to actually function.
- **`KEEP BUT PRUNE CLAIMS`**: Core mechanism is functional, but the surrounding theoretical narrative must be stripped down to reality.

---

## Complete Component Classification Matrix

| Component / Subsystem | Location | Current State / Behavior | Classification | Actionable Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Navier-Stokes Singularity Probe (`ns-probe`)** | [`src/experiment.rs:529-788`](file:///data/data/com.termux/files/home/projects/titan_text/src/experiment.rs#L529-L788), [`src/main.rs:579-756`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L579-L756) | Simulates "Millennium Prize Cases C & D" blowup on 1D grid; triggers on arbitrary threshold `enstrophy > 25.0` | **SAFE TO DELETE** | Delete `cmd_ns_probe`, `run_navier_stokes_blowup_probe`, and `Command::NsProbe`. |
| **Enstrophy, Palinstrophy & BKM Norm** | [`src/field.rs:195-233`](file:///data/data/com.termux/files/home/projects/titan_text/src/field.rs#L195-L233), [`src/train.rs:124-127`](file:///data/data/com.termux/files/home/projects/titan_text/src/train.rs#L124-L127) | Computes 3D hydrodynamic metrics on 1D lattice; evaluated every training epoch without backprop | **SAFE TO DELETE** | Remove from `field.rs` and purge from `train.rs:StepDiagnostics`. |
| **Spectral Cascade Analysis** | [`src/field.rs:238-331`](file:///data/data/com.termux/files/home/projects/titan_text/src/field.rs#L238-L331) | Computes enstrophy spectrum, power-law slope $\beta$, and dissipation scale via discrete Fourier sum | **SAFE TO DELETE** | 94 lines of unvectorized $O(B \cdot C \cdot L^2)$ DFT loops contributing zero causal learning signal. |
| **Laplacian Viscous Dissipation** | [`src/nca.rs:97-116`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L97-L116) | Discrete heat diffusion ($\nu \Delta \mathbf{x}$) with substepping to damp high-frequency noise | **KEEP BUT PRUNE CLAIMS** | Retain linear spatial smoothing; purge all claims of Navier-Stokes Millennium Prize singularity arrest. |
| **Dual-Timescale Slow/Fast Gating** | [`src/latent.rs:238-261`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L238-L261), [`src/config.rs:26`](file:///data/data/com.termux/files/home/projects/titan_text/src/config.rs#L26) | Masks updates for subset of channels on non-cadence ticks; defaults to 0.0; never used in training | **SAFE TO DELETE** | Delete `slow_channel_fraction` and `slow_timescale_cadence` across config, CLI, and latent executor. |
| **`periodic_boundary` Config Field** | [`src/config.rs:12`](file:///data/data/com.termux/files/home/projects/titan_text/src/config.rs#L12), [`src/field.rs:167`](file:///data/data/com.termux/files/home/projects/titan_text/src/field.rs#L167) | Config field serialized in manifests but NEVER read by any spatial stencil; rolls are unconditionally periodic | **SAFE TO DELETE** | Delete dead field from `FieldConfig` and manifest schema. |
| **`TitanConfig.model` Field** | [`src/config.rs:126, 213`](file:///data/data/com.termux/files/home/projects/titan_text/src/config.rs#L126), [`src/main.rs:931`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L931) | Config validates model name ("transformer", "gru", etc.), but training and sweeps only ever run NCA | **SAFE TO DELETE** | Remove field; Titan Text only trains NCA. |
| **`observation_interval` Config Field**| [`src/latent.rs:30`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L30) | Config field declared with default 1; never accessed in execution | **SAFE TO DELETE** | Remove dead config field. |
| **12 Dead CLI Flags on `rollout`** | [`src/cli.rs:76-88`](file:///data/data/com.termux/files/home/projects/titan_text/src/cli.rs#L76-L88), [`src/main.rs:757-830`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L757-L830) | `--latent-ticks`, `--pause-ticks`, `--slow-*`, `--lesion-*` are accepted by CLI parser but never read in `cmd_rollout` | **SAFE TO DELETE** | Remove 12 dead options from `cli.rs:options(Command::Rollout)`. |
| **Redundant `Command::Falsify`** | [`src/cli.rs:26, 42`](file:///data/data/com.termux/files/home/projects/titan_text/src/cli.rs#L26), [`src/main.rs:1661`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L1661) | Separate CLI command that literally executes `cmd_probe` identically | **SAFE TO DELETE** | Delete `Command::Falsify` enum variant; alias `falsify -> probe` in parser. |
| **`benchmark --epochs` Flag** | [`src/cli.rs:152`](file:///data/data/com.termux/files/home/projects/titan_text/src/cli.rs#L152), [`src/main.rs:1262`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L1262) | Parsed as `_epochs` and ignored; benchmarks do not train models | **SAFE TO DELETE** | Remove unused argument. |
| **Untrained Baseline Comparisons** | [`src/baselines.rs`](file:///data/data/com.termux/files/home/projects/titan_text/src/baselines.rs), [`src/main.rs:1258-1470`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L1258-L1470) | Compares NCA forward pass against randomly initialized Transformer/GRU/RNN with 0 optimizer steps | **SAFE TO DELETE** | Delete `cmd_benchmark` accuracy tables until real baseline training loops exist. |
| **`nca.develop_with_forcing`** | [`src/nca.rs:213-226`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L213-L226) | Dead helper annotated with `#[allow(dead_code)]` | **SAFE TO DELETE** | Delete unused method. |
| **`field.inject_damage`** | [`src/field.rs:146-164`](file:///data/data/com.termux/files/home/projects/titan_text/src/field.rs#L146-L164) | Dead helper annotated with `#[allow(dead_code)]` | **SAFE TO DELETE** | Delete unused method. |
| **`tasks.verify_task_suite_integrity`**| [`src/tasks.rs:666`](file:///data/data/com.termux/files/home/projects/titan_text/src/tasks.rs#L666) | Dead method causing active Cargo compiler warning | **SAFE TO DELETE** | Delete unused method. |
| **`tasks.generate_suite`** | [`src/tasks.rs:124`](file:///data/data/com.termux/files/home/projects/titan_text/src/tasks.rs#L124) | Dead method annotated with `#[allow(dead_code)]` | **SAFE TO DELETE** | Delete unused method. |
| **`baselines.SequenceModel::step`** | [`src/baselines.rs:11`](file:///data/data/com.termux/files/home/projects/titan_text/src/baselines.rs#L11) | Dead trait method annotated with `#[allow(dead_code)]` | **SAFE TO DELETE** | Delete unused trait method. |
| **`instrumentation.to_json/jsonl`** | [`src/instrumentation.rs:149, 155`](file:///data/data/com.termux/files/home/projects/titan_text/src/instrumentation.rs#L149) | Dead serialization methods annotated with `#[allow(dead_code)]` | **SAFE TO DELETE** | Delete unused methods. |
| **`instrumentation.aggregate_summaries`**| [`src/instrumentation.rs:248`](file:///data/data/com.termux/files/home/projects/titan_text/src/instrumentation.rs#L248) | Dead struct and aggregation helper annotated with `#[allow(dead_code)]` | **SAFE TO DELETE** | Delete unused struct and function. |
| **Masked Loss & Accuracy Functions** | [`src/vocab.rs:156, 172`](file:///data/data/com.termux/files/home/projects/titan_text/src/vocab.rs#L156), [`src/dataset.rs:99, 105`](file:///data/data/com.termux/files/home/projects/titan_text/src/dataset.rs#L99) | `masked_cross_entropy_loss` implemented but completely bypassed by `Trainer` and `LatentExecutor` | **CANDIDATE FOR SIMPLIFICATION** | Wire into `Trainer::train_step` or replace unmasked loss with unified `loss(logits, targets, mask: Option<&Tensor>)`. |
| **Nonlinear Perturbation Probe ($Q$ Norm)**| [`src/experiment.rs:175-256`](file:///data/data/com.termux/files/home/projects/titan_text/src/experiment.rs#L175-L256) | Claims high-frequency perturbations induce macro emergence; actually evaluates pointwise activation $f''(x)$ | **KEEP BUT PRUNE CLAIMS** | Retain as curvature check; prune pseudo-multiscale emergence claims. |
| **Attractor Basin Probe (`cmd_attractor`)**| [`src/latent.rs:575-637`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L575-L637) | Feeds two distinct prompt strings, checks $\|x_A - x_B\|_2 > 0.15\|x\|_2$, claims "hysteresis detected" | **KEEP BUT PRUNE CLAIMS** | Prune "hysteresis" and "attractor basin" narrative; it is simply input sensitivity. |
| **Memory as Dynamics Probe (`cmd_memory`)**| [`src/latent.rs:643-702`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L643-L702) | Compares $E_1 + E_2$ vs $E_2$ alone; checks if $\|x_{E_1+E_2} - x_{E_2}\|_2 > 0.05$ | **KEEP BUT PRUNE CLAIMS** | Prune theoretical framing; it simply tests recurrent state carryover. |
| **Dynamic Association Probe (`associate`)**| [`src/latent.rs:476-570`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L476-L570) | Probes target probability ramping; caused by state norm growth driving softmax temperature to 0 | **CANDIDATE FOR SIMPLIFICATION** | Replace unnormalized readout with normalized readout ($x / \|x\|_2$) to prevent false temperature collapse. |
| **Autonomous Trajectory Classification** | [`src/experiment.rs:339-350`](file:///data/data/com.termux/files/home/projects/titan_text/src/experiment.rs#L339-L350) | Classifies unrolls into "PERIODIC_LIMIT_CYCLE", "RECURRENT_BOUNDED_ORBIT", "OPEN_PHASE_DRIFT" | **CANDIDATE FOR SIMPLIFICATION** | Replace 40 lines of distance search with a 5-line norm growth and variance monitor. |

---

## Detailed Audit: Area by Area

### 1. Pseudo-Physics & The Navier-Stokes Singularity Delusion

#### What the Code Does
In [`src/field.rs:195-331`](file:///data/data/com.termux/files/home/projects/titan_text/src/field.rs#L195-L331) and [`src/experiment.rs:529-788`](file:///data/data/com.termux/files/home/projects/titan_text/src/experiment.rs#L529-L788), the codebase computes:
- Enstrophy: $\Omega = \frac{1}{2} \text{mean}(\|\nabla \mathbf{x}\|^2)$
- Palinstrophy: $P = \frac{1}{2} \text{mean}(\|\Delta \mathbf{x}\|^2)$
- BKM Norm: $\max_{b, i} \|\nabla \mathbf{x}_{b, i}\|_2$
- Discrete Fourier enstrophy spectra $\Omega(k) = (2\pi k / L)^2 E(k)$ and power-law slopes $\beta$
- Singularity detector: triggers blow-up when `enstrophy > 25.0` or `enstrophy > 100.0`
- CLI command `ns-probe` ([`src/main.rs:579-756`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L579-L756)) claiming to test "Cases C & D Fefferman formulation" and "Beale-Kato-Majda blow-up."

#### Why It Does Not Causally Earn Its Keep
1. **Mathematical Impossibility in 1D**:
   The Beale-Kato-Majda (1984) theorem applies strictly to 3D incompressible Euler equations. In 3D, vorticity $\boldsymbol{\omega} = \nabla \times \mathbf{u}$ evolves via:
   $$\frac{\partial \boldsymbol{\omega}}{\partial t} + (\mathbf{u} \cdot \nabla)\boldsymbol{\omega} = (\boldsymbol{\omega} \cdot \nabla)\mathbf{u} + \nu \Delta \boldsymbol{\omega}$$
   Singularities require the vortex stretching term $(\boldsymbol{\omega} \cdot \nabla)\mathbf{u}$ to amplify gradients to infinity. In a **1D discrete sequence lattice**:
   - The curl operator $\nabla \times$ does not exist.
   - Vortex stretching is identically zero.
   - Incompressibility ($\nabla \cdot \mathbf{u} = 0$) in 1D requires $\partial u / \partial x = 0$, forcing velocity to be spatially constant.
2. **Discrete Bounded Lipschitz Dynamics Cannot Blow Up**:
   In `nca.rs:step_with_forcing`, the state update is:
   $$\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha (\text{sigmoid}(W_g h) \odot \tanh(W_\delta h))$$
   Because $|\tanh| \le 1$ and $|\text{sigmoid}| \le 1$, the update delta is strictly bounded:
   $$\|\Delta \mathbf{x}\|_\infty \le \alpha = 0.5$$
   After $T$ steps, the maximum possible state norm is bounded by $\|\mathbf{x}_T\|_\infty \le \|\mathbf{x}_0\|_\infty + 0.5 T$. A strictly bounded linear ramp in discrete time cannot experience finite-time blow-up.
3. **Arbitrary Threshold Posed as Singularity**:
   In [`src/experiment.rs:580-588`](file:///data/data/com.termux/files/home/projects/titan_text/src/experiment.rs#L580-L588), `is_blowup` is triggered by `enstrophy > 25.0` or `vel > 80.0`. A sequence model whose activation norm exceeds 25.0 has simply experienced unregularized drift; calling this a Navier-Stokes singularity is pure narrative inflation.
4. **Training Compute Waste**:
   In [`src/train.rs:124-127`](file:///data/data/com.termux/files/home/projects/titan_text/src/train.rs#L124-L127), every single epoch computes:
   ```rust
   let enstrophy = field.enstrophy()?;
   let palinstrophy = field.palinstrophy()?;
   let bkm_norm = field.bkm_norm()?;
   let freq_decomp = field.spatial_frequency_decomposition()?;
   ```
   None of these values enter the loss function or backward pass. They add pure overhead to every training step on Termux CPU.

#### What Breaks if Deleted?
**Zero sequence modeling functionality breaks.** The entire `ns-probe` command, all 260 lines of fluid probe logic in `experiment.rs`, and the Fourier enstrophy calculations in `field.rs` can be purged without altering a single weight, gradient, or token prediction.

#### The Minimal Replacement
Retain linear Laplacian diffusion in `nca.rs:apply_viscous_dissipation` as a standard spatial smoothing regularizer (`viscosity`), but purge all hydrodynamic claims, BKM references, and diagnostic clutter.

---

### 2. Dual-Timescale Slow/Fast Gating vs Simple Homogeneous Updates

#### What the Code Does
[`src/latent.rs:238-261`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L238-L261) implements mixed-timescale channel execution:
```rust
let effective_delta = if latent_cfg.slow_channel_fraction > 0.0
    && latent_cfg.slow_timescale_cadence > 1
{
    let (_, _, c) = modified_x.dims3()?;
    let num_slow = ((c as f32) * latent_cfg.slow_channel_fraction.clamp(0.0, 1.0)).round() as usize;
    let is_slow_active = tick % latent_cfg.slow_timescale_cadence == 0;

    if !is_slow_active && num_slow > 0 {
        let mut mask_vec = vec![1.0f32; c];
        let slow_start = c.saturating_sub(num_slow);
        for ch in slow_start..c { mask_vec[ch] = 0.0; }
        let mask = Tensor::from_slice(&mask_vec, (1, 1, c), device)?;
        gated_delta.broadcast_mul(&mask)?
    } else {
        gated_delta
    }
} else {
    gated_delta
};
```

#### Why It Does Not Causally Earn Its Keep
1. **Never Used in Training**:
   In [`src/train.rs:93`](file:///data/data/com.termux/files/home/projects/titan_text/src/train.rs#L93), training calls `self.nca.step(&field, &self.device)`. The NCA step function does not have slow/fast channel gating; it updates all channels homogeneously on every step.
2. **Dead in All Checkpoints**:
   `slow_channel_fraction` defaults to `0.0` in `LatentConfig`, and all checkpoints (`v0_text`, `v0_viscous`, `falsify_dev1`, etc.) record `0.0`.
3. **Inference Degradation**:
   Freezing a subset of channels at inference when the network was trained with all channels active on every BPTT step breaks feature representations.
4. **Heap Allocation on Every Tick**:
   When active, it dynamically allocates a `Vec<f32>` and a Candle `Tensor::from_slice` on every non-cadence tick on the heap.

#### Recommendation
**SAFE TO DELETE**. Remove `slow_channel_fraction` and `slow_cadence` from `LatentConfig`, `config.rs`, `cli.rs`, and `latent.rs`.

---

### 3. Redundant CLI Flags, Dead Fields & Dead Methods

#### A. Dead Configuration Fields
1. **`FieldConfig.periodic_boundary`** ([`src/config.rs:12`](file:///data/data/com.termux/files/home/projects/titan_text/src/config.rs#L12)):
   - Declared as `bool`, serialized into every checkpoint manifest, validated in `config.rs`.
   - **Zero references in spatial code**: `nca.rs` and `field.rs` call `roll_spatial`, which unconditionally implements circular periodic boundaries. If a user sets `periodic_boundary = false`, the code ignores it.
   - **Verdict**: **SAFE TO DELETE**.
2. **`TitanConfig.model`** ([`src/config.rs:126, 213`](file:///data/data/com.termux/files/home/projects/titan_text/src/config.rs#L126)):
   - Validates that `model` is "nca", "transformer", "gru", etc.
   - Set in `main.rs:931` from `--model`.
   - **Never read again**: `train.rs` only trains NCA; `cmd_sweep` only evaluates NCA. Passing `--model transformer` to `sweep` silently evaluates NCA.
   - **Verdict**: **SAFE TO DELETE**.
3. **`LatentConfig.observation_interval`** ([`src/latent.rs:30`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L30)):
   - Never referenced in any execution loop.
   - **Verdict**: **SAFE TO DELETE**.
4. **`ProbeConfig.forcing_amplitude`** ([`src/config.rs:97`](file:///data/data/com.termux/files/home/projects/titan_text/src/config.rs#L97)):
   - Only used by the pseudo-physics `ns-probe`.
   - **Verdict**: **SAFE TO DELETE** along with `ns-probe`.

#### B. Dead CLI Flags
1. **12 Dead Flags on `Command::Rollout`**:
   - In [`src/cli.rs:76-88`](file:///data/data/com.termux/files/home/projects/titan_text/src/cli.rs#L76-L88), `Rollout` accepts:
     `--latent-ticks`, `--pause-ticks`, `--slow-cadence`, `--slow-fraction`, `--lesion-state`, `--lesion-gates`, `--lesion-residual`, `--lesion-gain`, `--lesion-noise`, `--lesion-freeze`, `--lesion-reset`, `--lesion-channels`.
   - In [`src/main.rs:757-900`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L757-L900), `cmd_rollout` **does not parse or read a single one of these flags**.
   - A user passing `--lesion-state` or `--slow-fraction 0.5` to `rollout` has their arguments silently discarded.
   - **Verdict**: **SAFE TO DELETE** from `cli.rs:options(Command::Rollout)`.
2. **Redundant `Command::Falsify`**:
   - [`src/cli.rs:26, 42`](file:///data/data/com.termux/files/home/projects/titan_text/src/cli.rs#L26) defines `falsify` as a distinct command. In [`src/main.rs:1661`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L1661), it is handled as:
     `cli::Command::Probe | cli::Command::Falsify => cmd_probe(&args, &device)`
   - It is an exact duplicate of `probe`.
   - **Verdict**: **SAFE TO DELETE** (merge into `probe` aliases).
3. **`benchmark --epochs`**:
   - Parsed as `let _epochs = args.value("--epochs")?` in `main.rs:1262` and discarded.
   - **Verdict**: **SAFE TO DELETE**.

#### C. Dead Methods Annotated with `#[allow(dead_code)]`
- [`src/tasks.rs:666`](file:///data/data/com.termux/files/home/projects/titan_text/src/tasks.rs#L666): `verify_task_suite_integrity` (causes Cargo compiler warning).
- [`src/tasks.rs:124`](file:///data/data/com.termux/files/home/projects/titan_text/src/tasks.rs#L124): `generate_suite`.
- [`src/nca.rs:213`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L213): `develop_with_forcing`.
- [`src/field.rs:146`](file:///data/data/com.termux/files/home/projects/titan_text/src/field.rs#L146): `inject_damage`.
- [`src/latent.rs:706`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L706): `run_latent_budget_sweep`.
- [`src/baselines.rs:11`](file:///data/data/com.termux/files/home/projects/titan_text/src/baselines.rs#L11): `SequenceModel::step`.
- [`src/instrumentation.rs:149, 155, 248`](file:///data/data/com.termux/files/home/projects/titan_text/src/instrumentation.rs#L149): `to_json`, `to_jsonl`, `aggregate_summaries`.
- **Verdict on all**: **SAFE TO DELETE**.

---

### 4. Over-Engineered State Probes & Narrative Inflation

#### A. Nonlinear Perturbation Probe ($Q$ Norm)
- **Code**: [`src/experiment.rs:175-256`](file:///data/data/com.termux/files/home/projects/titan_text/src/experiment.rs#L175-L256)
- **Claim**: *"Microscopic perturbations couple nonlinearly into macro-scale spatial structures."*
- **Mathematical Reality**:
  The probe injects $p_i = (-1)^i$ (alternating $\pm 1$) and evaluates the second difference:
  $$Q = \frac{\text{LP}(G(x + \epsilon p) + G(x - \epsilon p) - 2G(x))}{2\epsilon^2}$$
  For any smooth activation function $f(x)$, Taylor expansion yields:
  $$f(x + \epsilon p) + f(x - \epsilon p) - 2f(x) \approx \epsilon^2 p^2 f''(x)$$
  Because $p_i \in \{+1, -1\}$, $p_i^2 = (+1)^2 = (-1)^2 \equiv 1.0$ at every cell. The alternating high-frequency wave is converted into a **strictly constant (DC wavenumber 0) field** solely by the pointwise scalar activation! The low-pass filter ($\text{LP}$) preserves the DC mode.
  **This occurs even if the spatial stencil radius is zero** (no neighbor communication). It is a test of scalar activation curvature ($f''(x) \ne 0$), not multiscale spatial coupling.
- **Classification**: **KEEP BUT PRUNE CLAIMS** (prune the emergence narrative).

#### B. Attractor Basin & Hysteresis Probe (`cmd_attractor`)
- **Code**: [`src/latent.rs:575-637`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L575-L637)
- **Claim**: *"Tests convergence to competing state basins and hysteresis."*
- **Mathematical Reality**:
  The code runs forward passes for `prompt_a` and `prompt_b`, then tests if $\|x_A - x_B\|_2 > 0.15 \max(\|x_A\|, \|x_B\|)$. If true, it prints:
  `"SEPARATED_STABLE_REGIMES (hysteresis detected)"`
  Two different text strings producing different hidden states is the definition of a functioning neural network. Calling a non-zero Euclidean distance between distinct inputs "hysteresis" and "attractor basins" is meaningless.
- **Classification**: **CANDIDATE FOR SIMPLIFICATION / PRUNE CLAIMS**.

#### C. Memory as Dynamics Probe (`cmd_memory`)
- **Code**: [`src/latent.rs:643-702`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L643-L702)
- **Claim**: *"Tests memory as dynamics (earlier events persistently modifying state)."*
- **Mathematical Reality**:
  The code compares the state after processing $E_1 + E_2$ against processing $E_2$ alone from an empty field. If the distance exceeds 0.05, it claims "PERSISTENT_DYNAMIC_MEMORY." Any recurrent network has a hidden state that depends on prior tokens.
- **Classification**: **CANDIDATE FOR SIMPLIFICATION / PRUNE CLAIMS**.

#### D. Dynamic Concept Association Probe (`cmd_associate`)
- **Code**: [`src/latent.rs:476-570`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L476-L570)
- **Claim**: *"Semantic associations propagate continuously along a latent time axis."*
- **Mathematical Reality**:
  In `step_latent`, the state accumulates updates: $\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha \delta$. State norm $\|\mathbf{x}\|$ grows monotonically from $0.16$ to $>1.0$. Logits are computed via unnormalized projection $W_{out} \mathbf{x}$. As $\|\mathbf{x}\|$ inflates, logit magnitudes scale proportionally, driving softmax temperature toward zero ($\text{Softmax}(W \mathbf{x}) \to \text{one-hot}$).
  Negative controls reveal that feeding arbitrary stimuli ('z', 'q', or zero input) produces identical probability ramps to $>0.93$.
- **Classification**: **CANDIDATE FOR SIMPLIFICATION** (requires state normalization $\mathbf{x} / \|\mathbf{x}\|_2$ before readout to measure actual direction change rather than scalar norm growth).

#### E. Flawed Evaluation in `cmd_sweep`
- **Code**: [`src/latent.rs:777-779`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L777-L779)
- **Bug**:
  ```rust
  let flat_logits = logits.flatten_all()?;
  let probs = candle_nn::ops::softmax(&flat_logits, 0)?.to_vec1::<f32>()?;
  let mean_conf = probs.iter().fold(0.0f32, |m, &p| m.max(p));
  ```
  `logits` has shape `[batch, seq_len, vocab_size]`. Flattening all dimensions and running a 1D softmax means **all tokens across all sequence positions and batch items compete in a single softmax**. `mean_conf` is the maximum of an $N \times L \times V$ distribution, not the mean top-1 token confidence.
- **Arbitrary "Bifurcation" Detector** ([`src/latent.rs:815`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L815)):
  `let bifurcation = points.windows(2).any(|w| (w[1].accuracy - w[0].accuracy).abs() > 0.25);`
  Labeling an accuracy jump $\ge 25\%$ between two unroll depths as a "bifurcation" is an abuse of dynamical systems terminology.
- **Classification**: **CANDIDATE FOR SIMPLIFICATION**.

---

## The Minimalist Target Architecture: 5 Primitives

If Titan Text is stripped of every line of decorative baggage, what remains is an elegant, highly efficient cellular sequence model:

```
┌─────────────────────────────────────────────────────────────┐
│                    TITAN TEXT MINIMAL CORE                  │
├─────────────────────────────────────────────────────────────┤
│ 1. MorphogenicField: Tensor [B, L, C]                       │
│    - Energy: 0.5 * mean(x^2)                                │
│    - Distance: ||x - y||                                    │
│    - Stencils: roll_spatial(x, -1) and roll_spatial(x, 1)   │
├─────────────────────────────────────────────────────────────┤
│ 2. NeuralCellularAutomaton:                                 │
│    - Local Perception: Cat([Identity, Gradient, Laplacian]) │
│    - MLP: Linear -> GELU -> (Linear_delta, Linear_gate)     │
│    - Step: x + alpha * (sigmoid(gate) * tanh(delta))        │
│    - Smoothing: + (nu * alpha) * Laplacian(x)               │
├─────────────────────────────────────────────────────────────┤
│ 3. TokenInterface:                                          │
│    - Embedding [V, C]                                       │
│    - Readout Head [C, V]                                    │
├─────────────────────────────────────────────────────────────┤
│ 4. Trainer:                                                 │
│    - BPTT unroll for T steps                                │
│    - Masked Cross-Entropy Loss & AdamW                      │
├─────────────────────────────────────────────────────────────┤
│ 5. Tasks / Evaluator:                                       │
│    - Text next-token prediction                             │
│    - Algorithmic benchmarks with masked answer evaluation   │
└─────────────────────────────────────────────────────────────┘
```

### Lines of Code Reduction Potential

| Subsystem / Area | Current Lines | Minimal Replacement | Lines Saved |
| :--- | :--- | :--- | :--- |
| `src/field.rs` (Enstrophy, BKM, Fourier cascade) | 439 lines | 85 lines | **~354 lines** |
| `src/experiment.rs` (NS blowup probe, fluid traces) | 927 lines | 250 lines | **~677 lines** |
| `src/latent.rs` (Dual timescales, attractor/memory probes) | 987 lines | 350 lines | **~637 lines** |
| `src/main.rs` (`cmd_ns_probe`, duplicate CLI handlers) | 1,680 lines | 950 lines | **~730 lines** |
| `src/baselines.rs` (Untrained baseline mocks) | 397 lines | 0 lines (until trained) | **~397 lines** |
| `src/cli.rs` (Dead flags, duplicate commands) | 440 lines | 260 lines | **~180 lines** |
| **Total Codebase Reduction** | **~4,870 lines** | **~1,895 lines** | **~2,975 lines (-61%)** |

---

## Immediate Action Items for Engineering

1. **Delete `cmd_ns_probe` and `run_navier_stokes_blowup_probe`**:
   Remove all references to Millennium Prize, Beale-Kato-Majda, enstrophy, and palinstrophy. Keep `viscosity` strictly as discrete Laplacian heat smoothing.
2. **Purge Dual-Timescale Channels**:
   Delete `slow_channel_fraction` and `slow_timescale_cadence` across `config.rs`, `latent.rs`, `main.rs`, and `cli.rs`.
3. **Clean Up CLI and Config**:
   - Delete `periodic_boundary`, `model`, and `observation_interval` from configs.
   - Remove the 12 dead flags from `cli.rs:options(Command::Rollout)`.
   - Remove `Command::Falsify` and make `falsify` an alias for `probe`.
   - Remove `benchmark --epochs`.
4. **Delete Unused Methods**:
   Remove `verify_task_suite_integrity`, `generate_suite`, `develop_with_forcing`, `inject_damage`, `aggregate_summaries`, and `to_jsonl`.
5. **Fix the Evaluation Hygiene**:
   - Wire `loss_mask` from `TaskBatch` into the loss and accuracy calculations in `cmd_benchmark` and `cmd_sweep`.
   - Normalize state vectors before the readout head in `cmd_associate` ($\mathbf{x} / \|\mathbf{x}\|_2$).
   - Fix 2D batch/sequence softmax in `cmd_sweep`.
