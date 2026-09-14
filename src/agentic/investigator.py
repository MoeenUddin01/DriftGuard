"""AI Investigator core logic."""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Investigator:
    """AI Agent responsible for investigating feature/data drift incidents."""

    def investigate(self, payload: Any) -> Dict[str, Any]:
        """Process a DriftIncidentPayload and conduct root-cause investigation.
        
        Args:
            payload: DriftIncidentPayload instance containing incident details.
            
        Returns:
            Dict containing investigation status and details.
        """
        incident_id = getattr(payload, "incident_id", "unknown")
        drifted_features = getattr(payload, "drifted_features", [])
        
        logger.info(f"Investigator received drift alert [Incident: {incident_id}]")
        logger.info(f"Investigating {len(drifted_features)} drifted features: {drifted_features}")
        
        return {
            "status": "investigation_triggered",
            "incident_id": incident_id,
            "drifted_features": drifted_features,
        }
