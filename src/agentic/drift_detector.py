"""Statistical Drift Detector component.

Monitors incoming customer inference data, calculates statistical distribution shifts
against saved baseline distributions (dataset/baseline_stats.json), and triggers
the AI Investigator agent when significant feature/data drift is detected.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import uuid

import numpy as np
import pandas as pd
from scipy import stats

from src.agentic.investigator import Investigator
from src.utils.config import get_config_value

logger = logging.getLogger(__name__)


@dataclass
class DriftIncidentPayload:
    """Structured payload generated when statistical drift is detected."""
    incident_id: str
    timestamp: str
    drifted_features: List[str]
    drift_scores: Dict[str, Dict[str, Any]]
    batch_sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize payload to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class DriftDetector:
    """Statistical Drift Detector to identify data distribution shifts in incoming inference batches."""

    def __init__(
        self,
        baseline_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
        investigator: Optional[Any] = None,
        ks_p_threshold: float = 0.05,
        psi_threshold: float = 0.20,
    ):
        """Initialize DriftDetector.

        Args:
            baseline_source: File path to baseline_stats.json or dictionary of baseline stats.
                             Defaults to config baseline_stats_path or 'dataset/baseline_stats.json'.
            investigator: Investigator instance to call when drift is detected. Defaults to Investigator().
            ks_p_threshold: p-value threshold for Kolmogorov-Smirnov test (drift flagged if p < threshold).
            psi_threshold: PSI threshold (drift flagged if PSI > threshold).
        """
        self.ks_p_threshold = ks_p_threshold
        self.psi_threshold = psi_threshold
        self.investigator = investigator or Investigator()
        self.baseline_stats = self._load_baseline(baseline_source)

    def _load_baseline(
        self, baseline_source: Optional[Union[str, Path, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Load and parse baseline statistical profiles."""
        if isinstance(baseline_source, dict):
            return baseline_source

        if baseline_source is None:
            resolved_path = get_config_value(
                "data.baseline_stats_path", "dataset/baseline_stats.json"
            )
            baseline_source = Path(resolved_path)

        path = Path(baseline_source)
        if not path.exists():
            raise FileNotFoundError(f"Baseline statistics file not found at: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded baseline stats from {path} ({len(data.get('features', {}))} features)")
            return data
        except Exception as e:
            logger.error(f"Failed to read baseline statistics from {path}: {e}")
            raise ValueError(f"Invalid baseline statistics file {path}: {e}") from e

    def compute_numerical_ks(
        self, baseline_feature_stats: Dict[str, Any], batch_series: pd.Series
    ) -> Tuple[float, float]:
        """Compute Kolmogorov-Smirnov 2-sample test against reference baseline distribution.

        Returns:
            Tuple of (statistic, p_value)
        """
        clean_batch = batch_series.dropna().values
        if len(clean_batch) < 2:
            return 0.0, 1.0

        # Extract or reconstruct baseline reference sample array
        ref_sample = self._extract_reference_sample(baseline_feature_stats)
        if len(ref_sample) < 2:
            return 0.0, 1.0

        try:
            ks_result = stats.ks_2samp(ref_sample, clean_batch)
            return float(ks_result.statistic), float(ks_result.pvalue)
        except Exception as e:
            logger.warning(f"Error computing KS test: {e}")
            return 0.0, 1.0

    def _extract_reference_sample(
        self, baseline_feature_stats: Dict[str, Any]
    ) -> np.ndarray:
        """Extract fine-grained quantile samples or generate synthetic sample from baseline stats."""
        if "quantiles_100" in baseline_feature_stats:
            return np.array(baseline_feature_stats["quantiles_100"], dtype=float)
        elif "deciles" in baseline_feature_stats:
            return np.array(baseline_feature_stats["deciles"], dtype=float)
        else:
            # Reconstruct from standard percentiles
            p_min = baseline_feature_stats.get("min", 0.0)
            p25 = baseline_feature_stats.get("25%", p_min)
            p50 = baseline_feature_stats.get("50%", (p25 + p_min) / 2)
            p75 = baseline_feature_stats.get("75%", p50)
            p_max = baseline_feature_stats.get("max", p75)

            # Generate 100 sample points interpolating percentiles
            q_points = [0.0, 0.25, 0.50, 0.75, 1.0]
            val_points = [p_min, p25, p50, p75, p_max]
            grid = np.linspace(0.0, 1.0, 100)
            return np.interp(grid, q_points, val_points)

    def compute_numerical_psi(
        self, baseline_feature_stats: Dict[str, Any], batch_series: pd.Series, num_bins: int = 10, eps: float = 1e-4
    ) -> float:
        """Compute Population Stability Index (PSI) for a numerical feature."""
        clean_batch = batch_series.dropna().values
        if len(clean_batch) == 0:
            return 0.0

        # Get bin edges from deciles or range
        if "deciles" in baseline_feature_stats:
            raw_edges = np.array(baseline_feature_stats["deciles"], dtype=float)
        else:
            p_min = baseline_feature_stats.get("min", float(clean_batch.min()))
            p_max = baseline_feature_stats.get("max", float(clean_batch.max()))
            raw_edges = np.linspace(p_min, p_max, num_bins + 1)

        # Ensure strictly increasing bin edges
        edges = self._make_edges_unique(raw_edges)
        n_actual_bins = len(edges) - 1
        if n_actual_bins < 1:
            return 0.0

        expected_prop = np.ones(n_actual_bins) / n_actual_bins

        # Calculate actual counts in each bin
        counts, _ = np.histogram(clean_batch, bins=edges)
        actual_prop = counts / len(clean_batch)

        # PSI formula: sum((actual - expected) * ln((actual + eps) / (expected + eps)))
        psi_value = np.sum(
            (actual_prop - expected_prop) * np.log((actual_prop + eps) / (expected_prop + eps))
        )
        return float(max(0.0, psi_value))

    def _make_edges_unique(self, edges: np.ndarray) -> np.ndarray:
        """Add small perturbations to ensure bin edges are strictly increasing."""
        unique_edges = [edges[0]]
        for e in edges[1:]:
            if e <= unique_edges[-1]:
                unique_edges.append(unique_edges[-1] + 1e-5)
            else:
                unique_edges.append(e)
        return np.array(unique_edges, dtype=float)

    def compute_categorical_psi(
        self, baseline_feature_stats: Dict[str, Any], batch_series: pd.Series, eps: float = 1e-4
    ) -> float:
        """Compute Population Stability Index (PSI) for a categorical feature."""
        base_freqs: Dict[str, float] = baseline_feature_stats.get("frequencies", {})
        clean_batch = batch_series.dropna().astype(str)
        
        if len(clean_batch) == 0:
            return 0.0

        batch_counts = clean_batch.value_counts(normalize=True).to_dict()

        # All unique categories across baseline and batch
        all_categories = set(base_freqs.keys()).union(set(batch_counts.keys()))
        if not all_categories:
            return 0.0

        psi_total = 0.0
        for cat in all_categories:
            e_i = base_freqs.get(cat, eps)
            a_i = batch_counts.get(cat, 0.0)
            psi_total += (a_i - e_i) * np.log((a_i + eps) / (e_i + eps))

        return float(max(0.0, psi_total))

    def compute_categorical_chi2(
        self, baseline_feature_stats: Dict[str, Any], batch_series: pd.Series
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute Chi-Square goodness-of-fit test for categorical feature.

        Returns:
            Tuple of (chi2_stat, p_value) or (None, None) if sample size too small.
        """
        base_freqs: Dict[str, float] = baseline_feature_stats.get("frequencies", {})
        clean_batch = batch_series.dropna().astype(str)
        n_obs = len(clean_batch)

        if n_obs < 5 or not base_freqs:
            return None, None

        categories = list(base_freqs.keys())
        batch_counts = clean_batch.value_counts().to_dict()

        # Build observed and expected frequency lists
        obs_list = []
        exp_list = []

        unseen_count = 0
        for cat, val in batch_counts.items():
            if cat not in base_freqs:
                unseen_count += val

        for cat in categories:
            obs_list.append(batch_counts.get(cat, 0))
            exp_list.append(base_freqs[cat] * n_obs)

        # Include unseen category bucket if present
        if unseen_count > 0:
            obs_list.append(unseen_count)
            exp_list.append(1e-4 * n_obs)

        obs_arr = np.array(obs_list, dtype=float)
        exp_arr = np.array(exp_list, dtype=float)

        # Normalize expected frequencies to match sum of observed
        if exp_arr.sum() > 0:
            exp_arr = exp_arr * (obs_arr.sum() / exp_arr.sum())

        try:
            res = stats.chisquare(f_obs=obs_arr, f_exp=exp_arr)
            return float(res.statistic), float(res.pvalue)
        except Exception as e:
            logger.warning(f"Error computing Chi2 test: {e}")
            return None, None

    def detect(
        self, df: pd.DataFrame
    ) -> Tuple[bool, DriftIncidentPayload, Dict[str, Dict[str, Any]]]:
        """Detect statistical feature drift in an incoming customer data batch.

        Args:
            df: Input inference DataFrame containing customer features.

        Returns:
            Tuple of (has_drift, DriftIncidentPayload, drift_scores_dict)
        """
        logger.info(f"Running Statistical Drift Detector on batch of {len(df)} samples...")
        features_baseline = self.baseline_stats.get("features", {})
        drift_scores: Dict[str, Dict[str, Any]] = {}
        drifted_features: List[str] = []

        for col, col_baseline in features_baseline.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' present in baseline stats but missing in batch.")
                continue

            col_type = col_baseline.get("type", "numerical")
            batch_series = df[col]

            if col_type == "numerical":
                ks_stat, ks_p = self.compute_numerical_ks(col_baseline, batch_series)
                psi_val = self.compute_numerical_psi(col_baseline, batch_series)

                is_drifted = bool((ks_p < self.ks_p_threshold) or (psi_val > self.psi_threshold))

                drift_scores[col] = {
                    "type": "numerical",
                    "ks_stat": round(ks_stat, 5),
                    "ks_p_value": round(ks_p, 5),
                    "psi": round(psi_val, 5),
                    "is_drifted": is_drifted,
                }

            else:  # categorical
                psi_val = self.compute_categorical_psi(col_baseline, batch_series)
                chi2_stat, chi2_p = self.compute_categorical_chi2(col_baseline, batch_series)

                is_drifted = bool(psi_val > self.psi_threshold)
                if chi2_p is not None and chi2_p < self.ks_p_threshold:
                    is_drifted = True

                drift_scores[col] = {
                    "type": "categorical",
                    "psi": round(psi_val, 5),
                    "chi2_stat": round(chi2_stat, 5) if chi2_stat is not None else None,
                    "chi2_p_value": round(chi2_p, 5) if chi2_p is not None else None,
                    "is_drifted": is_drifted,
                }

            if is_drifted:
                drifted_features.append(col)

        has_drift = len(drifted_features) > 0
        incident_id = f"inc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        payload = DriftIncidentPayload(
            incident_id=incident_id,
            timestamp=timestamp,
            drifted_features=drifted_features,
            drift_scores=drift_scores,
            batch_sample_size=len(df),
        )

        if has_drift:
            logger.warning(
                f"DRIFT DETECTED in {len(drifted_features)} feature(s): {drifted_features}. "
                f"Dispatching payload to AI Investigator [ID: {incident_id}]."
            )
            try:
                self.investigator.investigate(payload)
            except Exception as e:
                logger.error(f"Error calling Investigator: {e}")
        else:
            logger.info(f"No feature drift detected across {len(df)} samples.")

        return has_drift, payload, drift_scores
