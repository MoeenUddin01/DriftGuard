"""AI Investigator Agent component.

Processes DriftIncidentPayload notifications, localizes drifted feature distributions,
evaluates model feature importances to estimate accuracy loss, reasons about root causes,
and formulates diagnostic engineering recommendations.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_config_value

logger = logging.getLogger(__name__)


class Investigator:
    """AI Agent responsible for investigating feature/data drift incidents."""

    def __init__(
        self,
        baseline_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
        model_source: Optional[Union[str, Path, Any]] = None,
    ):
        """Initialize Investigator agent.

        Args:
            baseline_source: Baseline stats file path or dictionary.
            model_source: Path to serialized model artifact or fitted model instance.
        """
        self.baseline_stats = self._load_baseline(baseline_source)
        self.model = self._load_model(model_source)

    def _load_baseline(
        self, baseline_source: Optional[Union[str, Path, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Load baseline stats dictionary or JSON file."""
        if isinstance(baseline_source, dict):
            return baseline_source

        if baseline_source is None:
            resolved_path = get_config_value(
                "data.baseline_stats_path", "dataset/baseline_stats.json"
            )
            baseline_source = Path(resolved_path)

        path = Path(baseline_source)
        if not path.exists():
            logger.warning(f"Baseline statistics file not found at: {path}. Using empty baseline.")
            return {"features": {}}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read baseline statistics from {path}: {e}")
            return {"features": {}}

    def _load_model(self, model_source: Optional[Union[str, Path, Any]]) -> Optional[Any]:
        """Load trained model instance or joblib artifact."""
        if model_source is None:
            resolved_path = get_config_value(
                "model.artifact_path", "models/model.joblib"
            )
            model_source = Path(resolved_path)

        if isinstance(model_source, (str, Path)):
            path = Path(model_source)
            if not path.exists():
                logger.warning(f"Model artifact not found at: {path}. Model importance weighting disabled.")
                return None
            try:
                model = joblib.load(path)
                logger.info(f"Loaded model artifact from {path}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
                return None
        return model_source

    def _get_feature_importances(self) -> Dict[str, float]:
        """Extract feature importances from model if available."""
        if self.model is None:
            return {}

        # 1. Direct dict or object attribute
        if hasattr(self.model, "get_config"):
            try:
                cfg = self.model.get_config()
                feature_names = cfg.get("feature_names", [])
                underlying = getattr(self.model, "model", None)
                if underlying and hasattr(underlying, "feature_importances_"):
                    importances = underlying.feature_importances_
                    if len(feature_names) == len(importances):
                        return dict(zip(feature_names, map(float, importances)))
            except Exception:
                pass

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_names = getattr(self.model, "feature_names_in_", [f"feature_{i}" for i in range(len(importances))])
            return dict(zip(feature_names, map(float, importances)))

        return {}

    def localize_shifts(
        self, payload: Any, batch_df: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Pinpoint drifted features, rank by severity, and quantify shift direction & magnitude.

        REQ-INV-001 implementation.
        """
        drifted_features = getattr(payload, "drifted_features", [])
        drift_scores = getattr(payload, "drift_scores", {})
        baseline_features = self.baseline_stats.get("features", {})

        localized: List[Dict[str, Any]] = []

        for feat in drifted_features:
            scores = drift_scores.get(feat, {})
            base_info = baseline_features.get(feat, {})
            col_type = scores.get("type", base_info.get("type", "numerical"))
            psi_val = scores.get("psi", 0.0)
            ks_stat = scores.get("ks_stat", 0.0)
            ks_p = scores.get("ks_p_value", 1.0)

            shift_info: Dict[str, Any] = {
                "feature": feat,
                "type": col_type,
                "psi": psi_val,
                "ks_stat": ks_stat,
                "ks_p_value": ks_p,
                "direction": "shifted",
                "pct_change": None,
                "summary": f"Feature '{feat}' drifted (PSI: {psi_val:.4f})",
            }

            # Quantify direction & magnitude if batch data is available
            if batch_df is not None and feat in batch_df.columns:
                series = batch_df[feat].dropna()
                if len(series) > 0:
                    if col_type == "numerical":
                        base_mean = base_info.get("mean", None)
                        batch_mean = float(series.mean())

                        if base_mean is not None and base_mean != 0:
                            pct = ((batch_mean - base_mean) / abs(base_mean)) * 100.0
                            shift_info["pct_change"] = round(pct, 2)
                            direction = "increased" if pct > 0 else "decreased"
                            shift_info["direction"] = direction
                            shift_info["summary"] = (
                                f"Numerical feature '{feat}' mean {direction} by {abs(pct):.1f}% "
                                f"(baseline: {base_mean:.2f}, batch: {batch_mean:.2f}, PSI: {psi_val:.4f})"
                            )
                        else:
                            shift_info["summary"] = f"Numerical feature '{feat}' mean: {batch_mean:.2f} (PSI: {psi_val:.4f})"
                    else:
                        base_freqs = base_info.get("frequencies", {})
                        batch_freqs = series.value_counts(normalize=True).to_dict()
                        top_batch_cat = max(batch_freqs, key=batch_freqs.get) if batch_freqs else "unknown"
                        top_batch_pct = batch_freqs.get(top_batch_cat, 0.0) * 100.0
                        shift_info["summary"] = (
                            f"Categorical feature '{feat}' distribution shifted. Top category: '{top_batch_cat}' "
                            f"({top_batch_pct:.1f}% of batch, PSI: {psi_val:.4f})"
                        )

            localized.append(shift_info)

        # Rank by severity (PSI primary, KS statistic secondary)
        localized.sort(key=lambda x: (x["psi"], x["ks_stat"]), reverse=True)
        return localized

    def analyze_root_cause(
        self, payload: Any, localized_shifts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate model importance weights, estimate performance degradation, and reason on root causes.

        REQ-INV-002 implementation.
        """
        importances = self._get_feature_importances()
        total_importance_weight = sum(importances.values()) if importances else 0.0

        drifted_features = [s["feature"] for s in localized_shifts]

        # Calculate importance-weighted drift severity
        weighted_drift_impact = 0.0
        feature_impacts: List[Dict[str, Any]] = []

        for shift in localized_shifts:
            feat = shift["feature"]
            psi_val = shift["psi"]
            importance = importances.get(feat, 0.0)

            # Impact score = PSI * Feature Importance Weight
            impact_score = psi_val * (importance / total_importance_weight if total_importance_weight > 0 else 0.1)
            weighted_drift_impact += impact_score

            feature_impacts.append({
                "feature": feat,
                "psi": psi_val,
                "importance_weight": round(importance, 4),
                "impact_score": round(impact_score, 4),
            })

        # Estimate ROC-AUC / F1 degradation
        # Empirical heuristic: estimated ROC-AUC drop = min(0.25, 0.15 * weighted_drift_impact)
        estimated_roc_auc_drop = round(min(0.25, 0.15 * max(0.1, weighted_drift_impact)), 4)

        # Generate root-cause hypotheses
        hypotheses: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        num_drifted = len(drifted_features)
        if num_drifted > 0:
            feat_list_str = ", ".join(drifted_features[:3])

            # Hypothesis 1: Macroeconomic / Demographic shift
            hypotheses.append({
                "hypothesis_id": "HYP-001",
                "category": "Macroeconomic & Demographic Shift",
                "confidence": "HIGH" if num_drifted >= 2 else "MEDIUM",
                "description": (
                    f"Significant distributional shift in key borrower features ({feat_list_str}) indicates "
                    "changing customer applicant demographics or shifting economic conditions."
                ),
            })

            # Hypothesis 2: Data Pipeline Ingestion Anomaly
            hypotheses.append({
                "hypothesis_id": "HYP-002",
                "category": "Data Ingestion Pipeline Anomaly",
                "confidence": "MEDIUM",
                "description": (
                    f"Possible upstream data pipeline schema change or unit conversion error affecting {feat_list_str}."
                ),
            })

            # Formulate engineering recommendations
            recommendations.append("Audit upstream data ingestion pipeline for unit conversions or missing feature values.")
            recommendations.append(f"Trigger retraining pipeline on recent customer window containing {num_drifted} drifted features.")
            recommendations.append("Adjust classification decision probability threshold to mitigate false default risk.")
        else:
            hypotheses.append({
                "hypothesis_id": "HYP-000",
                "category": "No Significant Drift",
                "confidence": "HIGH",
                "description": "No significant feature drift detected.",
            })

        return {
            "weighted_drift_impact": round(weighted_drift_impact, 4),
            "estimated_roc_auc_drop": estimated_roc_auc_drop,
            "feature_impacts": feature_impacts,
            "hypotheses": hypotheses,
            "recommended_actions": recommendations,
        }

    def investigate(
        self, payload: Any, batch_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Process a DriftIncidentPayload, perform shift localization, root cause reasoning, and impact estimation.

        Args:
            payload: DriftIncidentPayload instance containing incident metadata.
            batch_df: Optional DataFrame of incoming batch for exact shift calculation.

        Returns:
            Dict containing comprehensive investigation diagnostic findings.
        """
        incident_id = getattr(payload, "incident_id", "unknown")
        drifted_features = getattr(payload, "drifted_features", [])

        logger.info(f"AI Investigator analyzing incident {incident_id} ({len(drifted_features)} drifted features)...")

        # 1. Feature Localization & Shift Analysis (REQ-INV-001)
        localized_shifts = self.localize_shifts(payload, batch_df=batch_df)

        # 2. Root Cause Reasoning & Impact Estimation (REQ-INV-002)
        root_cause_analysis = self.analyze_root_cause(payload, localized_shifts)

        investigation_result = {
            "status": "investigation_completed",
            "incident_id": incident_id,
            "timestamp": getattr(payload, "timestamp", ""),
            "batch_sample_size": getattr(payload, "batch_sample_size", 0),
            "num_drifted_features": len(drifted_features),
            "ranked_localized_shifts": localized_shifts,
            "root_cause_analysis": root_cause_analysis,
        }

        logger.info(
            f"Investigation completed for incident {incident_id}. "
            f"Estimated ROC-AUC drop: -{root_cause_analysis['estimated_roc_auc_drop']:.4f}"
        )

        return investigation_result
