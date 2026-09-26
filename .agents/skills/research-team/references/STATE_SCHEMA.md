# Canonical Research-State Packet Schema (Version 3.0)

The research state packet (`.agents/state/research_packet.json`) prevents context amnesia, ensures high-context fidelity for DeepSeek collaborators, and guarantees that agents never reason from superseded historical artifacts.

---

## JSON Schema Specification

```json
{
  "topic": "String: Title of research campaign",
  "created_at_utc": "ISO8601 Timestamp",
  "updated_at_utc": "ISO8601 Timestamp",
  "confirmed_facts": [
    {
      "id": "F1",
      "statement": "String: Factual finding supported by quantitative measurement",
      "source_measurement": "String: Run ID, benchmark JSON, or log path",
      "timestamp": "ISO8601 Timestamp"
    }
  ],
  "supported_interpretations": [
    {
      "id": "SUP_1",
      "statement": "String: Interpretation that fits evidence but has plausible alternatives",
      "competing_alternatives": "String: Counter-explanations",
      "timestamp": "ISO8601 Timestamp"
    }
  ],
  "current_hypotheses": [
    {
      "id": "H1",
      "statement": "String: Testable mechanistic causal hypothesis",
      "confidence": 0.75,
      "status": "active | testing",
      "created_at_utc": "ISO8601 Timestamp"
    }
  ],
  "rejected_hypotheses": [
    {
      "id": "FAL_1",
      "statement": "String: Disproven hypothesis or superseded finding",
      "falsified_by": "String: Forensic audit, lesion test, or experiment that invalidated it",
      "status": "falsified | artifact_debunked",
      "is_artifact": true,
      "rejected_at_utc": "ISO8601 Timestamp"
    }
  ],
  "unresolved_questions": [
    {
      "id": "Q1",
      "question": "String: Specific open inquiry",
      "priority": "high | normal | low",
      "status": "open | resolved"
    }
  ],
  "known_confounds": [
    {
      "id": "CONF_1",
      "confound": "String: Confounding factor, generator leakage, or measurement flaw",
      "mitigation_or_weakness": "String: Negative control or stratification required",
      "timestamp": "ISO8601 Timestamp"
    }
  ],
  "next_candidate_experiments": [
    {
      "id": "DISC_1",
      "name": "String: Experiment title",
      "target_hypotheses": ["H1", "H2"],
      "protocol": "String: Precise causal intervention and measurement protocol",
      "timestamp": "ISO8601 Timestamp"
    }
  ],
  "important_experimental_results": [
    {
      "id": "EXP_001",
      "metric": "String: Metric name (e.g. validation_loss_step_16)",
      "value": 0.042,
      "control_value": 0.185,
      "notes": "String: Paired seeds 42, 43, 44",
      "timestamp": "ISO8601 Timestamp"
    }
  ],
  "agent_disagreements": [
    {
      "id": "D1",
      "claim_a": "String: Position A",
      "claim_b": "String: Position B",
      "agents": ["DeepSeek", "Gemini"],
      "discriminating_test": "String: What experiment makes this empirical rather than rhetorical?",
      "status": "unresolved | resolved"
    }
  ],
  "uncertainty_structure": {
    "implementation": "low",
    "conceptual": "high",
    "empirical_support": "partial"
  },
  "state_insufficiency_events": []
}
```

---

## Seven Canonical Tiers Mapped

1. **`ESTABLISHED`** $\to$ `confirmed_facts`
2. **`SUPPORTED BUT NOT ESTABLISHED`** $\to$ `supported_interpretations`
3. **`HYPOTHESES`** $\to$ `current_hypotheses`
4. **`FALSIFIED / SUPERSEDED`** $\to$ `rejected_hypotheses` (with `is_artifact: true/false`)
5. **`OPEN QUESTIONS`** $\to$ `unresolved_questions`
6. **`KNOWN CONFOUNDS`** $\to$ `known_confounds`
7. **`NEXT DISCRIMINATING EXPERIMENTS`** $\to$ `next_candidate_experiments`
