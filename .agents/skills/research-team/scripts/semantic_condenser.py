#!/usr/bin/env python3
"""Semantic Context Condenser for Titan Research OS.

Faithfully extracts and compresses the full canonical laboratory state
(Mechanistic Thread 1 pushdown dynamics and Generative Thread 2 ASCII recurrence)
into high-density progressive disclosure tiers (Tier 0-3) with exact anchors,
epistemic markers, role-based lenses, and LLM semantic validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def get_secret(name: str) -> str:
    """Read secret from private ~/.config/titan/secrets.env without leaking."""
    secrets_file = Path(os.path.expanduser("~/.config/titan/secrets.env"))
    if not secrets_file.is_file():
        return os.environ.get(name, "")
    try:
        for line in secrets_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                val = line.split("=", 1)[1].strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                return val
    except Exception:
        pass
    return os.environ.get(name, "")


def get_git_state() -> Dict[str, str]:
    """Extract exact git anchors."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True, timeout=5
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=str(REPO_ROOT), text=True, timeout=5
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True, timeout=5
        ).strip()
        dirty = "dirty" if status else "clean"
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as e:
        return {"commit": "unknown", "branch": "unknown", "dirty": f"err:{e}"}


def read_text_safe(path: Path, max_lines: int | None = None) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if max_lines and len(lines) > max_lines:
            return "\n".join(lines[:max_lines])
        return "\n".join(lines)
    except Exception:
        return ""


class SemanticCondenser:
    def __init__(self, root: Path | None = None):
        self.root = root or REPO_ROOT

    def build_tier0(self, role: str, target_question: str) -> str:
        """Tier 0: Minimal mission, core rules, and immediate target question."""
        git = get_git_state()
        lines = [
            "# TITAN RESEARCH OS: CANONICAL SCIENTIFIC PACKET",
            f"REPOSITORY: titan_text | BRANCH: {git['branch']} | COMMIT: {git['commit'][:8]} ({git['dirty']})",
            f"TARGET SPECIALIST ROLE: {role}",
            "",
            "## 1. CORE OPERATIONAL NON-NEGOTIABLES [CONF:MAX]",
            "- Evidence-First: Every claim requires empirical artifacts, run logs, or code verification.",
            "- Two Active Research Threads:",
            "  * Thread 1 (Mechanistic): IPPR, coordinate channels, formal Chomsky pushdown dynamics. Unresolved bulk stagnation at L=16.",
            "  * Thread 2 (Generative): Tiny recurrent NCA-style autoregressive ASCII generation.",
            "- Causal Ablations Mandatory: Latent tick sweep tau in {0,1,2,4,8,16}, intact vs zero-tick (tau=0) vs lesion-state vs lesion-gates.",
            "- External exports to /sdcard/Download/TitanText/. No raw checkpoints or large logs in git.",
            "",
            f"## 2. ACTIVE RESEARCH TARGET: {target_question}",
        ]
        return "\n".join(lines)

    def build_tier1(self, role: str) -> str:
        """Tier 1: Canonical established findings across Thread 1 & Thread 2 with exact numerical anchors."""
        lines = [
            "",
            "## 3. CANONICAL ESTABLISHED EVIDENCE (QUANTITATIVE REPRODUCIBLE ANCHORS) [TIER 1]",
            "### THREAD 1: FORMAL PUSHDOWN DYNAMICS & RECURRENT CAUSALITY",
            "- [SUPPORTED] Causal Recurrence Necessity: Ablating latent ticks (tau=0 / --lesion-state) collapses IteratedParity (L=16, B=32) accuracy from 53.28% +- 3.42% to 0.00% (+3.35 nats loss, t(4)=34.80, p=4.07e-6, paired Cohen's d=15.56; reversal Delta x -> -Delta x divergence 14.13 nats, d=-75.44). (Source: eval_n5_lesion_state.json)",
            "- [SUPPORTED] Compact Horizon Parity: 1D NCA learns sequential parity on L=8, tau=8 with identity gap G_identity = +18.8% +- 4.4% (Slot 0: 68.8%, Slot 1: 68.8% intact vs 50.0% shuffled, d=1.90). (Source: campaign_arm2_curriculum.json)",
            "- [SUPPORTED] Correlation Length Limit: Continuous 1D NCA exhibits correlation length xi approx 4-6 cells across 1000 epochs, curriculum, coordinate channels, and single-slot supervision. On L=16, tau in 16..48, Slot 0 solves at 71-86%, while interior slots 1 and 2 remain at chance (44-50%). Carry probe spans 1 chunk (4 cells, 91.4% MLP acc) but collapses across 2 chunks (8 cells, 50.8%), invariant to unroll depth. (Source: campaign_horizon_48.json)",
            "- [SUPPORTED] Explicit Carry Register (ECR): ECR (T=16, C_c=16, causal DAG) achieves 100.0% train and 76.6% validation on iterated parity L=16 (beating Transformer 55.5%, GRU 55.5%, baseline NCA 71.9%), and lifts iterated-sum validation from 27.3% to 70.3% (matching RNN 70.3%, beating Transformer 43.8%). (Source: ecr_campaign_final_report.md)",
            "- [SUPPORTED] Discrete STE Quantization: ste_sign eliminates continuous dissipation drift, achieving in-distribution 58.0% +- 1.4% vs 56.1% +- 2.3% at L=16 and holding stability at 50.5% +- 0.1% at L=64 where continuous baselines decay to 49.1% +- 2.8% and bistable potential collapses to 45.8% +- 7.3%. (Source: drift_mitigation_campaign.json)",
            "- [SUPPORTED] Chomsky Type-2 Pushdown on Dyck-4: CD-DV-NCA achieves 43.90% +- 1.61% exact bracket prediction on Dyck-4 at L=64 (beating Markov-4 ceiling 0.098% by 448x). At depth D=4, carry lesion causes a -12.50% drop (50.5% -> 38.0%) and carry scramble collapses to chance (26.0%). Continuous channels alone collapse to chance (26.9% approx 25.0%) at D >= 12. (Source: dyck_pushdown_benchmark_results.json)",
            "- [SUPPORTED] Ballistic Horizon Law: Information roundtrip transport strictly requires horizon T* >= 2 ceil(L/k). At L=64, T=24, k=4, causal reach is 48 cells (75%), placing theoretical accuracy ceiling at 46.08% for D >= 12 where intact and lesion models collapse to chance (26-29%). (Source: stratified_depth_benchmark_results.json)",
            "- [SUPPORTED] Translation Invariance: Continuous field norm ||H|| exhibits bit-level invariance (Delta ||H|| <= 0.0023, p < 10^-6) under roll_spatial_4 across t* in {8,12,16} and depths D in {4,8}. Channel permutation perturbation Delta=0.35 at t*=8 monotonically contracts to Delta <= 0.05 at t*=16 as recurrence approaches fixed point. (Source: transport_conjugation_benchmark_results.json)",
            "- [SUPPORTED] Subspace Normalization (OCPD): Restricting bounded norm to continuous subspace H (bounded_h_only, ch 0..31) cuts loss by 47.3% at L=64 (5.13 -> 2.70) and 71.1% at L=128 (12.45 -> 3.59) while preserving 30.9% bracket accuracy. Full-channel uniform norm (Arm C) collapses discrete carry accuracy to chance (25.8% approx 25.0%). (Source: subspace_normalization_ocpd_results.json)",
            "- [SUPPORTED] Double-Dissociation Across Distance: Distance d = G + 2 + 2s on Dyck-4 (overall bracket pred 36.2%-58.3% across G in {0,4,8}, D in {2,4,8}) reveals double dissociation: local (d <= 4) solved by continuous H (85.4%-93.8%, minimal carry delta +4.1%), while non-local (d in [6..24]) requires discrete C_c (+14.6% to +25.0% causal gain over continuous chance floor 24.7%). (Source: adversarial_balanced_dyck_benchmark_results.json)",

            "",
            "### THREAD 2: AUTOREGRESSIVE GENERATIVE ASCII & ADAPTIVE HALTING",
            "- [SUPPORTED] Autoregressive ASCII NCA: 43,844 params, 64 channels, hidden dim 96, causal stencil N(i)={i-1, i} at tau in {4,8} generates structured multiline ASCII boxes (63.5% horizontal symmetry, 0/96 memorization, edit sim 0.391 vs 0.081 untrained). (Source: ascii_generative_campaign_report.md)",
            "- [SUPPORTED] Velocity Halting Sweet Spot: Relative velocity halting (delta <= theta, patience 2) at theta*=0.25 operates at mean tau = 4.53, achieving edit similarity 0.6719 (+27.4% over fixed tau=4 at 0.5274, exceeding fixed tau=8 at 0.6315 while saving 43.4% compute). (Source: ascii_adaptive_halting_campaign.md)",
            "- [OBS] Velocity vs Entropy Independence: Token predictive entropy H(P_t) is uncorrelated with stopping depth tau_t (r = -0.1475, N=684). Halting reflects internal dynamical convergence, not logit uncertainty.",
            "- [FALSIFY] Primary Mechanism: Budget-operating efficiency (avoiding under-deliberation at tau <= 2 and contractive over-smoothing collapse at tau=16). On held-out 10 seeds (N=30), Pareto gain is +10.71% (0.5535 vs 0.4735 at tau=4), while position-shuffled schedule Arm C narrows to Delta = +0.0292 (p = 0.1468).",
        ]
        return "\n".join(lines)

    def build_tier2(self, role: str) -> str:
        """Tier 2: Outstanding research debt and hypotheses under test."""
        claim_file = self.root / "reports" / "live_claim_ledger.md"
        debt_file = self.root / "reports" / "research_debt_ledger.md"
        claims = read_text_safe(claim_file, 60)
        debts = read_text_safe(debt_file, 50)

        lines = [
            "",
            "## 4. ACTIVE HYPOTHESES & UNRESOLVED RESEARCH DEBT [TIER 2]",
            "### HYPOTHESES AWAITING DECISIVE EMPIRICAL RESOLUTION",
            "- [HYP:H-UNITARY] Unitary / Skew-Symmetric Recurrence: Continuous orthogonal/unitary recurrence can bypass continuous dissipation without discrete STE quantization.",
            "- [HYP:H-TOPOLOGICAL] Soliton Pushdown Dynamics: Non-local transport in C_c is topologically protected by domain-wall phase boundaries.",
            "- [HYP:H-CHOP] Carrier-Modulation Orthogonality: In state x = [H; C_c], H acts as syntactic clock and C_c as non-local LIFO stack.",
            "",
            "### ACTIVE RESEARCH DEBT & METHODOLOGICAL CONTROLS",
            "- Debt 1: L=16 Bulk Interior Stagnation (Continuous NCA still cannot span >6 cells in unquantized 1D field).",
            "- Debt 2: Adaptive Halting Held-Out Shuffled Delta (Statistical significance of position-specific allocation vs pure budget distribution is p=0.1468).",
            "- Debt 3: Random-Compute Matched Sham (Ensuring adaptive gains are not reproducible by uniform random variable compute at identical mean tau).",
        ]

        if claims:
            lines.append("\n### Key Claim Ledger Entries:")
            for line in claims.splitlines():
                if line.strip().startswith("- Claim") or "Status:" in line:
                    lines.append(f"  {line.strip()}")

        return "\n".join(lines)

    def build_tier3(self, role: str) -> str:
        """Tier 3: Role-specific specialized deep context."""
        lines = [
            "",
            f"## 5. SPECIALIZED LENS FOR {role.upper()} [TIER 3]",
        ]

        if role == "code-review":
            lines.extend([
                "### Implementation Invariants to Audit:",
                "- Correctness of tau_min and tau_max clamping per token.",
                "- Verify state delta calculation: delta = ||x_t - x_{t-1}||_2 / (||x_t||_2 + eps).",
                "- Verify no hidden state leakages between independent samples or prompts.",
                "- Ensure lesion flags (--lesion-state, --lesion-gates) completely bypass intended channels.",
                "- Verify deterministic PRNG seeding with standard battery (42, 101, 202, 303, 404).",
            ])
        elif role == "falsification-arbiter":
            lines.extend([
                "### Adversarial Falsification Questions:",
                "- Could the adaptive compute advantage be explained entirely by whitespace early-exit?",
                "- Does the model allocate higher tau to structural corners/newlines or is allocation noise?",
                "- Is the comparison against fixed tau=8 fair given that fixed tau=8 suffers from contractive over-smoothing?",
                "- Are negative samples and failed seed rollouts preserved without cherry-picking?",
            ])
        elif role == "experiment-designer":
            lines.extend([
                "### Controlled Experiment Protocol Controls:",
                "- Matched compute control: Arm B fixed tau=4.5 (interpolated) or closest integers.",
                "- Permuted schedule control: Arm C (exact multiset of ticks shuffled across sequence).",
                "- Random variable compute control: Arm D (Uniform random tau in [1, 16] matched to mean tau).",
                "- Minimum sample battery: 10 unseen seeds, >=30 sequences per condition.",
            ])
        elif role == "dynamics-agent":
            lines.extend([
                "### Dynamical Measurements to Track:",
                "- Per-token tick correlation with token type (newline vs border vs interior vs whitespace).",
                "- State trajectory velocity ||x_t - x_{t-1}|| decay curve over recurrent ticks.",
                "- Mutual information between recurrent depth tau_t and preceding context entropy H(P_t).",
            ])

        return "\n".join(lines)

    def condense(
        self,
        role: str,
        target_question: str,
        max_tier: int = 2,
        budget_tokens: int = 2500,
    ) -> str:
        """Produce a consolidated, token-budgeted semantic packet."""
        parts = []
        parts.append(self.build_tier0(role, target_question))

        if max_tier >= 1:
            parts.append(self.build_tier1(role))

        if max_tier >= 2:
            parts.append(self.build_tier2(role))

        if max_tier >= 3:
            parts.append(self.build_tier3(role))

        full_text = "\n".join(parts)

        # Approximate token truncation (~3.8 chars per token)
        char_budget = int(budget_tokens * 3.8)
        if len(full_text) > char_budget:
            full_text = full_text[:char_budget] + "\n... [TRUNCATED TO FIT TOKEN BUDGET]"

        return full_text


def validate_condensation_semantic(
    raw_text: str,
    condensed_text: str,
    model: str = "deepseek/deepseek-v4.1-flash",
) -> Dict[str, Any]:
    """Validate semantic fidelity using OpenRouter via native httpx client."""
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not found in secrets.env"}

    import httpx

    prompt = f"""You are an adversarial Scientific Review Auditor for the Titan Text laboratory.
Compare the following RAW RESEARCH CONTEXT with the CONDENSED RESEARCH CONTEXT.
Verify whether any critical empirical numbers, active hypotheses, falsification criteria, or causal assertions from either Thread 1 (Mechanistic pushdown / IPPR) or Thread 2 (ASCII adaptive halting) have been lost, distorted, or hallucinated.

RAW CONTEXT (Length {len(raw_text)} chars):
---
{raw_text[:7000]}
---

CONDENSED CONTEXT (Length {len(condensed_text)} chars):
---
{condensed_text}
---

Return a strictly valid JSON object with:
{{
  "fidelity_score": <integer 0-100>,
  "missing_anchors": [<list of critical numbers or IDs dropped>],
  "distorted_claims": [<list of claims whose epistemic certainty was wrongly altered>],
  "compression_ratio": <float rounded to 2 decimals, len(condensed)/len(raw)>,
  "verdict": "<ACCEPTABLE|NEEDS_REVISION|REJECTED>",
  "summary": "<1-2 sentence critique explaining fidelity and semantic preservation>"
}}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Markon101/titan_text",
        "X-Title": "Titan Semantic Condenser",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=35.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return {"raw_response": content}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Titan Semantic Context Condenser")
    parser.add_argument("--role", default="falsification-arbiter", help="Target specialist role")
    parser.add_argument("--target", default="Evaluate adaptive halting dynamics and verify reproducibility", help="Core research question")
    parser.add_argument("--tier", type=int, default=3, choices=[0, 1, 2, 3], help="Max progressive tier")
    parser.add_argument("--budget", type=int, default=2500, help="Token budget cap")
    parser.add_argument("--out", help="Optional output path")
    parser.add_argument("--validate", action="store_true", help="Run live semantic validation via DeepSeek")

    args = parser.parse_args()

    condenser = SemanticCondenser()
    condensed = condenser.condense(
        role=args.role,
        target_question=args.target,
        max_tier=args.tier,
        budget_tokens=args.budget,
    )

    if args.out:
        Path(args.out).write_text(condensed, encoding="utf-8")
        print(f"Wrote condensed packet ({len(condensed)} chars) to {args.out}")
    else:
        print(condensed)

    if args.validate:
        print("\n=== Running Live Semantic Validation via DeepSeek ===")
        raw_context = (
            read_text_safe(REPO_ROOT / "docs" / "RESEARCH_STATE_PACKET.md")
            + "\n\n"
            + read_text_safe(REPO_ROOT / "reports" / "ascii_adaptive_halting_campaign.md")
        )
        audit = validate_condensation_semantic(raw_context, condensed)
        print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
