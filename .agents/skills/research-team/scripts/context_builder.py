#!/usr/bin/env python3
"""Rich Context Builder for DeepSeek Research Collaborators.

Constructs comprehensive, uncompressed, high-density scientific context packets
for DeepSeek 4.1 Flash research collaborators. Token economy is strictly secondary
to scientific correctness, independent reconstruction, and discovery.

Gathers up to 21 materially relevant dimensions of the Titan Text campaign:
1. Current research objective
2. Architecture and recurrent dynamics
3. Task definitions and why chosen
4. Training protocols
5. Latent tick semantics
6. Stabilization mechanisms
7. Current and historical metrics
8. Per-slot & stratified depth results
9. Activation/state diagnostics
10. Causal interventions & implementation
11. Statistical protocol
12. Previous hypotheses
13. Falsified hypotheses
14. Results later discovered to be artifacts
15. Current unresolved interpretations
16. Important source-code excerpts
17. Relevant documentation
18. Experiment reports
19. Activation traces / summary diagnostics
20. Git history / diffs when useful
21. Known limitations, suspected confounds & other agent arguments
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

# Base repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def get_git_diff(max_lines: int = 150) -> str:
    """Retrieve recent git diff or commit summary."""
    try:
        res = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        diff_stat = res.stdout.strip()
        log_res = subprocess.run(
            ["git", "log", "-n", "5", "--oneline"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        recent_commits = log_res.stdout.strip()
        return f"Recent Commits:\n{recent_commits}\n\nWorking Tree Stat:\n{diff_stat}"
    except Exception as e:
        return f"(Git info unavailable: {e})"


def read_file_safe(path: Path, max_lines: int | None = None) -> str:
    if not path.is_file():
        return f"(File not found: {path})"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if max_lines and len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} lines truncated]"
        return "\n".join(lines)
    except Exception as e:
        return f"(Error reading {path}: {e})"


class ContextBuilder:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or REPO_ROOT
        self.state_file = self.root / ".agents" / "state" / "research_packet.json"
        self.claim_ledger = self.root / "reports" / "live_claim_ledger.md"
        self.debt_ledger = self.root / "reports" / "research_debt_ledger.md"
        self.doc_state = self.root / "docs" / "RESEARCH_STATE_PACKET.md"

    def build_packet(
        self,
        objective: str,
        *,
        include_code: bool = True,
        include_reports: bool = True,
        include_git: bool = True,
        include_artifacts: bool = True,
        other_agent_arguments: str = "",
        custom_excerpts: list[str] | None = None,
    ) -> str:
        """Assemble the long, detailed canonical research context packet."""
        sections: list[str] = []

        # Header
        timestamp = datetime.now(timezone.utc).isoformat()
        sections.append(
            f"================================================================================\n"
            f"TITAN TEXT: HIGH-CONTEXT SCIENTIFIC RESEARCH PACKET\n"
            f"Generated: {timestamp}\n"
            f"Campaign: Recurrent Neural-Cellular Latent Dynamics & Formal Pushdown Computation\n"
            f"================================================================================\n"
        )

        # 1. Current Research Objective
        sections.append(
            f"### 1. CURRENT RESEARCH OBJECTIVE\n"
            f"{objective.strip()}\n"
        )

        # 2. Architecture & Recurrent Dynamics
        arch_desc = (
            "Model Family: Causal Continuous-Dynamics Discrete-Variable Neural Cellular Automata (CD-DV-NCA).\n"
            "- Lattice: 1D grid of length L cells with channels C (e.g. C=64, C=96).\n"
            "- Partitioned State: Direct-sum decomposition x = [H; C_c] where:\n"
            "    * H (channels 0..31): Continuous latent field with nonlinear residual MLP updates.\n"
            "    * C_c (channels 32..63): Discrete carry/stack registers with Straight-Through Estimator (STE sign).\n"
            "- Perception: Causal stencil DAG with skip strides (e.g. k=4 or k=1, 2) and causal mask.\n"
            "- Recurrent Dynamics: At tick tau, x_{t+1} = x_t + Delta x(Perception(x_t), x_t).\n"
            "    Local stencil computes neighbor differences; dense layers project to update vector.\n"
            "- Readout: Linear classification head projected from target cell positions at terminal tick T.\n"
        )
        sections.append(f"### 2. ARCHITECTURE & RECURRENT DYNAMICS\n{arch_desc}\n")

        # 3. Task Definitions & Why Chosen
        tasks_desc = (
            "Core Benchmarks:\n"
            "1. Iterated Parity (L=8, L=16, L=64): Cumulative XOR over sequential slots. Tests non-linear parity carry.\n"
            "2. Iterated Sum Dense (L=16): Multi-digit modulo accumulation. Tests dense arithmetic accumulation.\n"
            "3. Column Arithmetic (L=16): Multi-digit addition with reverse carry ripples. Decouples left/right directionality.\n"
            "4. Dyck-4 Pushdown (L=16, L=64, L=128): Chomsky Type-2 Context-Free Language of 4 balanced bracket pairs\n"
            "   ('()', '[]', '{}', '<>'). Evaluates non-local LIFO pushdown stack memory beyond finite Markov ceilings.\n"
            "Why chosen: These tasks strictly require non-local sequential information transport that feedforward\n"
            "local stencils cannot solve without genuine recurrence over latent ticks.\n"
        )
        sections.append(f"### 3. TASK DEFINITIONS & SELECTION RATIONALE\n{tasks_desc}\n")

        # 4. Latent Tick Semantics & Lightcone Kinematics
        tick_desc = (
            "- Latent Ticks (tau in [1..T]): Iterations of recurrent field updates on fixed token inputs.\n"
            "- Kinematic Lightcone Law: Information travels at most stride k per tick. Roundtrip transport\n"
            "  distance requires horizon T* >= 2 * ceil(L / k).\n"
            "- At L=64, k=4, T=24: Causal reach is 48 cells (75%), capping deep bracket resolution at D >= 12.\n"
            "- Over-unrolling risk: Models unrolled to T >> T_train without contractive regularization develop\n"
            "  continuous Lyapunov drift (loss explodes monotonically e.g. 2.28 -> 16.73 at L=128).\n"
        )
        sections.append(f"### 4. LATENT TICK SEMANTICS & HORIZON KINEMATICS\n{tick_desc}\n")

        # 5. Stabilization Mechanisms
        stab_desc = (
            "Stabilization Machinery:\n"
            "1. STE Sign Quantization: C_c = sign(tanh(w_carry * x + b)) with Straight-Through Estimator gradient.\n"
            "   Prevents continuous dissipation, enabling ballistic, dissipation-free non-local carry transport.\n"
            "2. Orthogonal Contractive-Pushdown Decomposition (OCPD) Subspace Normalization (Arm B: bounded_h_only):\n"
            "   Restricts norm bounding strictly to continuous channels H (0..31), damping Lyapunov growth by 47% at L=64\n"
            "   and 71% at L=128 while leaving discrete carry channels C_c unattenuated.\n"
            "   (Full-channel norm, Arm C, squashes discrete carry amplitudes, collapsing accuracy to uniform chance 25.8%).\n"
        )
        sections.append(f"### 5. STABILIZATION MECHANISMS\n{stab_desc}\n")

        # 6. Canonical Scientific State Packet (Established, Hypotheses, Falsified, Confounders, Discriminating Tests)
        if self.doc_state.is_file():
            state_text = read_file_safe(self.doc_state)
            sections.append(f"### 6. CANONICAL RESEARCH STATE DOCUMENTATION\n{state_text}\n")
        elif self.claim_ledger.is_file():
            ledger_text = read_file_safe(self.claim_ledger)
            sections.append(f"### 6. LIVE SCIENTIFIC CLAIM LEDGER\n{ledger_text}\n")

        # 7. Historical Failures, Artifacts & Falsifications (Explicitly Labeled)
        if include_artifacts:
            artifact_desc = (
                "CRITICAL WARNING: AGENTS MUST NEVER UNKNOWINGLY REASON FROM THESE SUPERSEDED RESULTS!\n"
                "Historical Artifacts & Falsified Claims:\n"
                "1. [ARTIFACT DEBUNKED] 'Canonical Baselines Collapse to 0.0% on Dyck-4':\n"
                "   Original benchmark evaluated untrained random weights due to an uninitialized checkpoint flag.\n"
                "   When properly trained with Pre-LN and gate retention, Simple RNN achieves 68.4%, Transformer 60.6%,\n"
                "   GRU 52.8% at L=16. Baselines only degrade under 4x length extrapolation (L=64: 46-52%).\n"
                "2. [ARTIFACT RESOLVED] 'FC-4 Carry Lesion Failure (No Causal Delta)':\n"
                "   Aggregate lesion delta appeared zero due to shallow bracket averaging (D=1 brackets resolved by local bigrams).\n"
                "   When stratified by depth, D=4 exhibits a decisive +12.5% causal carry delta (50.5% intact vs 38.0% lesion).\n"
                "3. [FALSIFIED] 'H1: Settling Latency Explains L=16 Failure':\n"
                "   Deep unroll to tau=48 with N=5 seeds yielded flat 46.9% accuracy; failure was spatial correlation length (xi ~ 4.5),\n"
                "   not settling time.\n"
                "4. [FALSIFIED] 'H2: 1D NCA Parity Impossibility':\n"
                "   Arm 2 Stage 1 achieved G_identity = +18.8% on L=8, proving continuous CAs can compute instance-specific parity.\n"
                "5. [SUPERSEDED] 'Theorem 12: CD-DV-NCA is Formally a DPDA':\n"
                "   Refuted by formal methods audit: finite grid with finite precision is mathematically a Chomsky Type-3 Finite State\n"
                "   Automaton. Claim is narrowed to Bounded DPDA Emulation up to capacity C_stack.\n"
            )
            sections.append(f"### 7. HISTORICAL ARTIFACTS & FALSIFIED HYPOTHESES (SUPERSEDED LEDGER)\n{artifact_desc}\n")

        # 8. Unresolved Interpretations & Suspected Confounds
        confounds_desc = (
            "Current Unresolved Scientific Interpretations & Confounds:\n"
            "1. Continuous Gating vs Discrete Quantization: Does ballistic transport strictly require STE quantization,\n"
            "   or could a unitary/skew-symmetric continuous operator achieve identical dissipation-free propagation?\n"
            "2. Spectral Damping Capacity Tradeoff: Arm B (bounded_h_only) cuts loss by 71% at L=128 but drops accuracy\n"
            "   from 36.7% to 30.9%. Is OCPD an asymptotic fixed point or merely an energy dissipation damper?\n"
            "3. Shallow Bracket Domination: In Dyck-4, 50% of brackets have nesting depth D <= 2, allowing local n-gram\n"
            "   heuristics to achieve ~40% accuracy without genuine non-local stack computation.\n"
            "4. Batch Shuffling Logit Temperature Artifact: Does cross-batch carry scrambling collapse predictions to 18.1%\n"
            "   due to true topological phase mismatch, or due to high logit variance saturating wrong classes?\n"
        )
        sections.append(f"### 8. UNRESOLVED INTERPRETATIONS & SUSPECTED CONFOUNDS\n{confounds_desc}\n")

        # 9. Key Source Code Excerpts
        if include_code:
            code_parts = []
            nca_path = self.root / "src" / "nca.rs"
            if nca_path.is_file():
                # Read key stencil lines
                code_parts.append(f"--- src/nca.rs (Excerpt: Causal Step & Perception) ---\n" + read_file_safe(nca_path, max_lines=120))
            latent_path = self.root / "src" / "latent.rs"
            if latent_path.is_file():
                code_parts.append(f"--- src/latent.rs (Excerpt: Field Partitioning & Subspace Normalization) ---\n" + read_file_safe(latent_path, max_lines=120))
            intervention_path = self.root / "src" / "intervention.rs"
            if intervention_path.is_file():
                code_parts.append(f"--- src/intervention.rs (Excerpt: Lesions, Transplants, Scrambling) ---\n" + read_file_safe(intervention_path, max_lines=120))
            if custom_excerpts:
                for ce in custom_excerpts:
                    p = self.root / ce
                    code_parts.append(f"--- {ce} ---\n" + read_file_safe(p, max_lines=100))
            sections.append(f"### 9. PRIMARY SOURCE CODE EXCERPTS\n" + "\n\n".join(code_parts) + "\n")

        # 10. Key Recent Empirical Reports
        if include_reports:
            reports_summary = []
            rep_files = [
                ("reports/stratified_depth_benchmark_results.json", "Stratified Depth Benchmark (Dyck-4 D=1..16)"),
                ("reports/transport_conjugation_benchmark_results.json", "Transport Conjugation Invariance Benchmark"),
                ("reports/subspace_normalization_ocpd_results.json", "Subspace Normalization (OCPD) 4-Arm Sweep"),
                ("reports/ood_128_causal_reach_sweep_results.json", "Extreme OOD L=128 Causal Reach Sweep"),
                ("reports/hostile_baselines_benchmark_results.json", "Hostile Baselines Benchmark (Transformer, GRU, RNN)"),
            ]
            for rpath, title in rep_files:
                p = self.root / rpath
                if p.is_file():
                    content = read_file_safe(p, max_lines=45)
                    reports_summary.append(f"[{title} ({rpath})]\n{content}\n")
            if reports_summary:
                sections.append("### 10. RECENT EMPIRICAL BENCHMARK REPORTS\n" + "\n".join(reports_summary))

        # 11. Git History & Recent Diffs
        if include_git:
            git_info = get_git_diff()
            sections.append(f"### 11. RECENT GIT HISTORY & CODE DRIFT\n{git_info}\n")

        # 12. Arguments From Other Agents (if any)
        if other_agent_arguments:
            sections.append(f"### 12. COMPETING AGENT ARGUMENTS & POSITIONS\n{other_agent_arguments.strip()}\n")

        sections.append(
            "================================================================================\n"
            "END OF RESEARCH CONTEXT PACKET. RECONSTRUCT INDEPENDENTLY BEFORE ADVISING.\n"
            "================================================================================\n"
        )

        return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Titan Text High-Context Packet Builder")
    parser.add_argument("objective", help="Current research objective or query")
    parser.add_argument("--out", help="Output file path (default: stdout)")
    parser.add_argument("--no-code", action="store_true", help="Exclude source code excerpts")
    parser.add_argument("--no-reports", action="store_true", help="Exclude report excerpts")
    parser.add_argument("--include-file", action="append", dest="custom_files", default=[], help="Include custom file path")

    args = parser.parse_args(argv)
    builder = ContextBuilder()
    packet = builder.build_packet(
        args.objective,
        include_code=not args.no_code,
        include_reports=not args.no_reports,
        custom_excerpts=args.custom_files,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(packet, encoding="utf-8")
        print(f"Context packet written to {out_path} ({len(packet)} chars, {len(packet.encode('utf-8'))} bytes)")
    else:
        print(packet)

    return 0


if __name__ == "__main__":
    sys.exit(main())
