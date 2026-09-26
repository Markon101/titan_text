#!/usr/bin/env python3
"""Timeseries and Trajectory Prediction Sensor.

Implements Part 14 of the Cognitive Control Architecture:
- Non-causal trajectory sensor designed to forecast diminishing returns,
  rework likelihood, uncertainty convergence, and context growth.
- Ready for pluggable backend forecasting models (e.g. TimesFM, ARIMA, or EWMA regression).
- Provides quantitative sensor signals to the Cognitive Governor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

from abc import ABC, abstractmethod
from telemetry import read_events, get_log_path


class BaseTrajectoryForecaster(ABC):
    """Abstract interface for time-series trajectory forecasters."""

    @abstractmethod
    def forecast(self, series: list[float], horizon: int = 1) -> dict[str, Any]:
        """Produce point forecast, drift slope, and confidence metrics."""
        pass


class HeuristicTrajectoryForecaster(BaseTrajectoryForecaster):
    """Deterministic linear-drift & moving-average forecaster."""

    def forecast(self, series: list[float], horizon: int = 1) -> dict[str, Any]:
        if not series:
            return {"mean": 0.0, "drift": 0.0, "volatility": 0.0, "last_value": 0.0, "point_forecast": 0.0}
        n = len(series)
        mean_val = sum(series) / n
        if n == 1:
            return {"mean": mean_val, "drift": 0.0, "volatility": 0.0, "last_value": series[0], "point_forecast": series[0]}

        x_mean = (n - 1) / 2.0
        denom = sum((i - x_mean) ** 2 for i in range(n))
        cov = sum((i - x_mean) * (series[i] - mean_val) for i in range(n))
        drift = cov / denom if denom > 0 else 0.0
        variance = sum((s - mean_val) ** 2 for s in series) / n
        volatility = math.sqrt(variance)

        point_forecast = round(series[-1] + (drift * horizon), 4)
        return {
            "mean": round(mean_val, 4),
            "drift": round(drift, 4),
            "volatility": round(volatility, 4),
            "last_value": series[-1],
            "point_forecast": point_forecast,
            "status": "ok",
        }


class TimesFMAdapter(BaseTrajectoryForecaster):
    """Pluggable adapter for Google's TimesFM zero-shot foundation model.

    Explicitly handles environments where TimesFM PyTorch/JAX runtime is absent.
    """

    def __init__(self, model_checkpoint: str = "google/timesfm-1.0-200m") -> None:
        self.model_checkpoint = model_checkpoint
        self.backend = None
        try:
            import timesfm  # type: ignore
            self.backend = timesfm.TimesFm(context_len=512, horizon_len=32)
        except (ImportError, Exception):
            self.backend = None

    def is_available(self) -> bool:
        return self.backend is not None

    def forecast(self, series: list[float], horizon: int = 1) -> dict[str, Any]:
        if not self.is_available():
            fallback_res = HeuristicTrajectoryForecaster().forecast(series, horizon=horizon)
            fallback_res["adapter"] = "timesfm_adapter_stub"
            fallback_res["model_present"] = False
            fallback_res["note"] = "TimesFM library not installed; degraded to heuristic baseline"
            return fallback_res

        try:
            preds = self.backend.forecast(series, freq=[0])
            point = float(preds[0][0])
            return {
                "point_forecast": point,
                "adapter": "timesfm_live",
                "model_present": True,
                "model_checkpoint": self.model_checkpoint,
                "status": "ok",
            }
        except Exception as e:
            fallback_res = HeuristicTrajectoryForecaster().forecast(series, horizon=horizon)
            fallback_res["adapter"] = "timesfm_error_fallback"
            fallback_res["error"] = str(e)
            return fallback_res


class TrajectorySensor:
    def __init__(
        self,
        log_path: Path | None = None,
        forecaster: BaseTrajectoryForecaster | None = None,
    ) -> None:
        self.log_path = log_path or get_log_path()
        self.events = read_events(self.log_path)
        self.forecaster = forecaster or HeuristicTrajectoryForecaster()

    def extract_time_series(self, metric: str = "rework_needed", max_points: int = 50) -> list[float]:
        """Extract a numeric time-series sequence from historical task events."""
        series: list[float] = []
        for e in self.events[-max_points:]:
            if metric == "rework_needed":
                series.append(1.0 if e.get("rework_needed") else 0.0)
            elif metric == "cost":
                series.append(float(e.get("total_cost_usd", 0.0)))
            elif metric == "elapsed_seconds":
                series.append(float(e.get("total_elapsed_seconds", 0.0)))
            elif metric == "reviewer_yield":
                series.append(1.0 if e.get("reviewer_found_problem") else 0.0)
            elif metric == "jev_calls":
                series.append(float(e.get("jev_calls_count", 0)))
            elif metric in e:
                try:
                    series.append(float(e[metric]))
                except (ValueError, TypeError):
                    pass
        return series

    def compute_trend(self, series: list[float]) -> dict[str, float]:
        """Fit a simple rolling exponential-weighted moving average and linear drift."""
        return self.forecaster.forecast(series, horizon=1)

    def predict_diminishing_returns(
        self,
        consecutive_passes: int,
        marginal_gain_history: list[float] | None = None,
    ) -> dict[str, Any]:
        """Predict whether additional reasoning passes have negligible expected information value."""
        if marginal_gain_history and len(marginal_gain_history) >= 2:
            trend = self.compute_trend(marginal_gain_history)
            prob_diminishing = 0.95 if (trend["last_value"] == 0.0 and trend["drift"] <= 0.0) else 0.40
        else:
            # Empirical sigmoid curve: after 2 passes without new evidence, probability rises sharply
            prob_diminishing = 1.0 / (1.0 + math.exp(-1.8 * (consecutive_passes - 2.0)))

        return {
            "prob_diminishing_returns": round(prob_diminishing, 3),
            "recommended_action": "stop_or_act" if prob_diminishing >= 0.70 else "continue_if_needed",
            "consecutive_passes": consecutive_passes,
        }

    def predict_likely_rework(self, task_category: str, thinking_budget: str) -> dict[str, Any]:
        """Predict probability of rework based on historical trajectory in this category/budget."""
        matching = [
            e for e in self.events
            if e.get("category") == task_category and e.get("thinking_budget") == thinking_budget
        ]
        if len(matching) < 3:
            return {
                "prob_rework": 0.15,
                "confidence": "low_sample_size",
                "sample_size": len(matching),
            }

        rework_rate = sum(1 for e in matching if e.get("rework_needed")) / len(matching)
        return {
            "prob_rework": round(rework_rate, 3),
            "confidence": "calibrated" if len(matching) >= 10 else "moderate",
            "sample_size": len(matching),
        }

    def predict_branch_explosion(self, active_branch_count: int, expansion_rate: float = 2.0) -> dict[str, Any]:
        """Forecast branch frontier growth and alert if width exceeds cognitive beam capacity."""
        forecast_next_step = active_branch_count * expansion_rate
        needs_pruning = forecast_next_step > 8 or active_branch_count >= 5
        return {
            "current_active_branches": active_branch_count,
            "forecast_next_step": round(forecast_next_step, 1),
            "needs_pruning_gate": needs_pruning,
            "recommended_max_width": 4,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trajectory Sensor CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # trend
    trend_p = subparsers.add_parser("trend", help="Analyze trend of historical telemetry metric")
    trend_p.add_argument("metric", default="rework_needed", choices=["rework_needed", "cost", "elapsed_seconds", "reviewer_yield", "jev_calls"])

    # rework
    rework_p = subparsers.add_parser("rework-prob", help="Predict rework risk for a task category & budget")
    rework_p.add_argument("category", help="Task category (e.g. coding, research)")
    rework_p.add_argument("budget", help="Thinking budget (e.g. minimal, normal, deep)")

    args = parser.parse_args(argv)
    sensor = TrajectorySensor()

    if args.subcommand == "trend":
        series = sensor.extract_time_series(args.metric)
        trend = sensor.compute_trend(series)
        print(f"=== TRAJECTORY TREND: {args.metric.upper()} ({len(series)} points) ===")
        print(json.dumps(trend, indent=2))

    elif args.subcommand == "rework-prob":
        pred = sensor.predict_likely_rework(args.category, args.budget)
        print(f"=== REWORK RISK PREDICTION: {args.category} / {args.budget} ===")
        print(json.dumps(pred, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
