"""Investigation Report Generator component.

Compiles diagnostic findings from the AI Investigator into structured, actionable
Markdown Investigation Reports and persists them to reports/drift_report_<timestamp>.md.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.utils.config import get_config_value

logger = logging.getLogger(__name__)


def save_report(
    markdown_content: str, output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Module-level convenience function to write a Markdown report to disk.

    Args:
        markdown_content: Rendered Markdown string.
        output_path: Optional destination file path. If None, auto-generates timestamp filename.

    Returns:
        Path: Path object pointing to written report file.
    """
    generator = ReportGenerator()
    return generator.save_report(markdown_content, output_path=output_path)


class ReportGenerator:
    """Compiles and exports structured Markdown investigation reports."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize ReportGenerator.

        Args:
            output_dir: Directory path where generated reports will be stored.
                        Defaults to config 'reports.output_dir' or 'reports/'.
        """
        if output_dir is None:
            resolved_dir = get_config_value("reports.output_dir", "reports")
            output_dir = Path(resolved_dir)

        self.output_dir = Path(output_dir)

    def determine_severity(self, num_drifted: int, roc_auc_drop: float) -> str:
        """Determine alert severity level based on drift magnitude and estimated ROC-AUC loss."""
        if num_drifted >= 3 or roc_auc_drop >= 0.10:
            return "CRITICAL"
        elif num_drifted >= 1 or roc_auc_drop >= 0.03:
            return "HIGH"
        elif num_drifted > 0:
            return "MODERATE"
        return "LOW"

    def generate_report(self, investigation_results: Dict[str, Any]) -> str:
        """Render investigation diagnostic findings into a formatted Markdown report.

        REQ-REP-001 implementation.

        Args:
            investigation_results: Dictionary output from Investigator.investigate().

        Returns:
            str: Validated Markdown document string.
        """
        incident_id = investigation_results.get("incident_id", "UNKNOWN_INCIDENT")
        timestamp = investigation_results.get(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )
        sample_size = investigation_results.get("batch_sample_size", 0)
        num_drifted = investigation_results.get("num_drifted_features", 0)

        rca = investigation_results.get("root_cause_analysis", {})
        roc_auc_drop = rca.get("estimated_roc_auc_drop", 0.0)
        severity = self.determine_severity(num_drifted, roc_auc_drop)

        localized_shifts = investigation_results.get("ranked_localized_shifts", [])
        hypotheses = rca.get("hypotheses", [])
        recommended_actions = rca.get("recommended_actions", [])
        feature_impacts = rca.get("feature_impacts", [])

        # Build Markdown Document
        lines: List[str] = []

        # Document Header
        lines.append(f"# 🚨 DriftGuard Investigation Report — `{incident_id}`")
        lines.append("")
        lines.append(f"> **Generated:** {timestamp}  ")
        lines.append(f"> **System:** Loan Default Risk Classification ML Pipeline")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 1. Executive Summary
        lines.append("## 1. Executive Summary")
        lines.append("")
        lines.append("| Metric | Details |")
        lines.append("|---|---|")
        lines.append(f"| **Incident ID** | `{incident_id}` |")
        lines.append(f"| **Timestamp** | `{timestamp}` |")
        lines.append(f"| **Alert Severity** | **`{severity}`** |")
        lines.append(f"| **Batch Sample Size** | `{sample_size}` customer applications |")
        lines.append(f"| **Drifted Features** | `{num_drifted}` feature(s) flagged |")
        lines.append(f"| **Est. ROC-AUC Degradation** | `-{roc_auc_drop:.4f}` |")
        lines.append("")

        if num_drifted > 0:
            lines.append(
                f"**Summary:** Statistical distribution drift was detected across **{num_drifted}** feature(s) "
                f"in an incoming inference batch of **{sample_size}** samples. The overall alert level is set to "
                f"**{severity}**. Immediate engineering review is recommended."
            )
        else:
            lines.append(
                "**Summary:** Baseline comparison complete. No significant feature drift detected in the current inference batch."
            )
        lines.append("")
        lines.append("---")
        lines.append("")

        # 2. Drifted Features Analysis
        lines.append("## 2. Drifted Features Analysis")
        lines.append("")

        if localized_shifts:
            lines.append("| Rank | Feature Name | Feature Type | Shift Direction | PSI Metric | KS Statistic | $p$-value |")
            lines.append("|---|---|---|---|---|---|---|")

            for rank, shift in enumerate(localized_shifts, 1):
                feat = shift.get("feature", "unknown")
                ftype = shift.get("type", "numerical")
                direction = shift.get("direction", "shifted")
                pct = shift.get("pct_change")
                psi_val = shift.get("psi", 0.0)
                ks_stat = shift.get("ks_stat", 0.0)
                ks_p = shift.get("ks_p_value", 1.0)

                dir_str = direction
                if pct is not None:
                    dir_str += f" ({pct:+.1f}%)"

                lines.append(
                    f"| #{rank} | `{feat}` | `{ftype}` | `{dir_str}` | `{psi_val:.4f}` | `{ks_stat:.4f}` | `{ks_p:.4f}` |"
                )
            lines.append("")

            lines.append("### Feature Shift Summaries")
            lines.append("")
            for shift in localized_shifts:
                summary_text = shift.get("summary", "")
                lines.append(f"- **`{shift.get('feature')}`**: {summary_text}")
            lines.append("")
        else:
            lines.append("No features exceeded statistical drift thresholds ($\text{PSI} > 0.20$ or $p_{\text{KS}} < 0.05$).")
            lines.append("")

        lines.append("---")
        lines.append("")

        # 3. Root Cause Analysis
        lines.append("## 3. Root Cause Analysis")
        lines.append("")

        if hypotheses:
            lines.append("The AI Investigator formulated the following diagnostic hypotheses:")
            lines.append("")
            for hyp in hypotheses:
                hyp_id = hyp.get("hypothesis_id", "HYP")
                category = hyp.get("category", "General")
                confidence = hyp.get("confidence", "MEDIUM")
                desc = hyp.get("description", "")
                lines.append(f"### [{hyp_id}] {category} (Confidence: `{confidence}`)")
                lines.append(f"{desc}")
                lines.append("")
        else:
            lines.append("No specific root cause hypotheses formulated.")
            lines.append("")

        lines.append("---")
        lines.append("")

        # 4. Performance & Financial Impact
        lines.append("## 4. Performance & Financial Impact")
        lines.append("")
        lines.append(f"- **Estimated Model ROC-AUC Degradation:** `-{roc_auc_drop:.4f}`")
        lines.append(f"- **Weighted Drift Impact Score:** `{rca.get('weighted_drift_impact', 0.0):.4f}`")
        lines.append("")

        if feature_impacts:
            lines.append("### Feature Importance Weighting Matrix")
            lines.append("")
            lines.append("| Feature | PSI Score | Importance Weight | Weighted Impact Score |")
            lines.append("|---|---|---|---|")
            for fi in feature_impacts:
                lines.append(
                    f"| `{fi.get('feature')}` | `{fi.get('psi', 0.0):.4f}` | `{fi.get('importance_weight', 0.0):.4f}` | `{fi.get('impact_score', 0.0):.4f}` |"
                )
            lines.append("")

        lines.append(
            "**Financial Risk Assessment:** Feature distribution shifts in loan applicant attributes increase the risk of misclassifying default probabilities. "
            "Unchecked drift may increase non-performing loan (NPL) exposure or cause unwarranted loan rejections."
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        # 5. Recommended Engineering Actions
        lines.append("## 5. Recommended Engineering Actions")
        lines.append("")

        if recommended_actions:
            for idx, action in enumerate(recommended_actions, 1):
                lines.append(f"{idx}. {action}")
            lines.append("")
        else:
            lines.append("1. Continue standard real-time statistical monitoring.")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*Report generated automatically by DriftGuard Agent System.*")

        return "\n".join(lines)

    def save_report(
        self,
        markdown_content: str,
        timestamp_str: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Write rendered Markdown report content to disk.

        REQ-REP-002 implementation.

        Args:
            markdown_content: Rendered Markdown text string.
            timestamp_str: Optional timestamp string for filename.
            output_path: Optional explicit file path destination.

        Returns:
            Path: Path to saved report file.
        """
        if output_path is not None:
            target_path = Path(output_path)
        else:
            ts = timestamp_str or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            target_path = self.output_dir / f"drift_report_{ts}.md"

        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"Successfully saved Investigation Report to {target_path}")
            return target_path.resolve()
        except Exception as e:
            logger.error(f"Failed to write investigation report to {target_path}: {e}")
            raise IOError(f"Could not save report to {target_path}: {e}") from e
