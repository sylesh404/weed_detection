"""Crop Risk Assessment Engine for AgriSight AI.

Evaluates potential weed-related crop risk to cotton based on:
- Weed abundance and spatial distribution
- Crop growth stage (Critical Period of Weed Competition: 15–60 Days After Sowing)
- Crop-weed proximity and root-zone interference
- Environmental context and previous field interventions

Categories: LOW, MODERATE, HIGH, INSUFFICIENT_DATA.
Maintains configurable thresholds with documented agronomical validation requirements.
Never fabricates yield loss percentages without empirical trial data.
"""
from typing import Dict, List, Optional, Any


class CropRiskEngine:
    """Assess potential weed competition risks to cotton crops using agronomic rules."""

    # Crop Growth Stages and Competition Vulnerability
    # The first 15 to 60 Days After Sowing (DAS) represents the Critical Period of Weed Competition (CPWC)
    # in cotton (ICAR-CICR / TNAU Agronomy Guidelines).
    CROP_STAGE_VULNERABILITY = {
        "EMERGENCE_SEEDLING": {
            "vulnerability": "CRITICAL",
            "cpwc_days": "15-35 DAS",
            "description": "Cotton seedlings are poor competitors; weed interference severely stunts root development.",
            "requires_validation": True,
        },
        "VEGETATIVE_SQUARING": {
            "vulnerability": "HIGH",
            "cpwc_days": "30-60 DAS",
            "description": "Active vegetative growth and flower bud initiation; weed competition restricts sympodial branching.",
            "requires_validation": True,
        },
        "FLOWERING_BOLL_DEV": {
            "vulnerability": "MODERATE",
            "cpwc_days": "60-90 DAS",
            "description": "Canopy begins closing; weeds compete for nutrients, moisture, and harbor insect pests.",
            "requires_validation": True,
        },
        "BOLL_OPENING_MATURITY": {
            "vulnerability": "LOW",
            "cpwc_days": ">90 DAS",
            "description": "Canopy shaded; weeds primarily risk trash contamination during harvest rather than vegetative competition.",
            "requires_validation": True,
        },
        "UNKNOWN": {
            "vulnerability": "UNCERTAIN",
            "cpwc_days": "Unknown",
            "description": "Growth stage not provided; risk cannot be evaluated against critical competition windows.",
            "requires_validation": False,
        },
    }

    # Configurable Evaluation Parameters
    DEFAULT_CONFIG = {
        "high_risk_weed_count": 8,
        "moderate_risk_weed_count": 3,
        "high_risk_crop_proximity_count": 2,
        "require_expert_validation_for_treatment": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def evaluate_risk(
        self,
        weed_analysis: Dict[str, Any],
        spatial_analysis: Dict[str, Any],
        field_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assess potential crop risk from weed presence and context.

        Args:
            weed_analysis: Output from WeedAnalyzer.
            spatial_analysis: Output from SpatialAnalyzer.
            field_context: Optional agricultural metadata (growth_stage, days_after_sowing,
                           recent_rainfall, past_herbicide, etc.).

        Returns:
            Risk assessment dictionary.
        """
        ctx = field_context or {}
        counts = weed_analysis.get("counts", {})
        data_qual = weed_analysis.get("data_quality", {})
        dispersion = weed_analysis.get("dispersion", {})
        proximity = spatial_analysis.get("inter_crop_proximity", {})

        visible_weeds = counts.get("visible_weed_count", 0)
        visible_cotton = counts.get("visible_cotton_count", 0)
        is_concentrated = dispersion.get("is_concentrated", False)
        adjacent_weeds = proximity.get("weeds_adjacent_to_crop", 0)
        crop_weed_overlap = proximity.get("crop_weed_overlap_detected", False)

        missing_info: List[str] = []
        evidence: List[str] = []

        # 1. Check for Insufficient Data
        if visible_weeds == 0 and visible_cotton == 0:
            return {
                "risk_category": "LOW",
                "risk_explanation": "No weeds or crop plants detected in the evaluated frame.",
                "evidence": ["Zero weed detections present"],
                "missing_information": ["Field visual confirmation"],
                "assessment_priority": "ROUTINE_MONITORING",
                "requires_field_verification": False,
                "scientific_limitation_note": (
                    "Risk classification reflects observable frame evidence only and does not predict "
                    "yield losses without field trials and certified agronomist calibration."
                ),
            }

        # Check optical reliability
        if not data_qual.get("reliable_optical_evidence", False) and visible_weeds > 0:
            missing_info.append("Higher optical detection confidence (current confidence is below 40%)")

        # 2. Check Crop Growth Stage
        stage_input = str(ctx.get("growth_stage", "UNKNOWN")).upper().replace(" ", "_")
        matched_stage = "UNKNOWN"
        for key in self.CROP_STAGE_VULNERABILITY:
            if key in stage_input or stage_input in key:
                matched_stage = key
                break

        stage_info = self.CROP_STAGE_VULNERABILITY.get(
            matched_stage, self.CROP_STAGE_VULNERABILITY["UNKNOWN"]
        )

        if matched_stage == "UNKNOWN":
            missing_info.append(
                "Crop growth stage / Days After Sowing (DAS) to determine Critical Period of Weed Competition"
            )
        else:
            evidence.append(
                f"Crop stage identified as '{matched_stage}' (vulnerability: {stage_info['vulnerability']})"
            )

        # 3. Check Weed Species Specificity
        weed_species = ctx.get("weed_species")
        if not weed_species:
            missing_info.append(
                "Specific weed taxonomy/species (current model classifies generic grass/weed)"
            )
        else:
            evidence.append(f"Recorded weed species: {weed_species}")

        # 4. Check Prior Management
        past_action = ctx.get("recent_management_action")
        if past_action:
            evidence.append(f"Prior field action: {past_action}")

        # 5. Core Decision Logic for Risk Category
        # Insufficient data condition: optical confidence poor and stage unknown
        if (
            not data_qual.get("reliable_optical_evidence", False)
            and matched_stage == "UNKNOWN"
            and visible_weeds > 0
        ):
            risk_category = "INSUFFICIENT_DATA"
            explanation = (
                "Insufficient evidence to assess crop risk reliably. Low optical detection confidence "
                "combined with unknown crop growth stage prevents calculating competition threat."
            )
            priority = "VERIFICATION_REQUIRED"
            requires_field_verification = True

        elif visible_weeds == 0:
            risk_category = "LOW"
            explanation = "Field frame shows active cotton foliage without detectable weed interference."
            priority = "ROUTINE_MONITORING"
            requires_field_verification = False

        elif (
            (visible_weeds >= self.config["high_risk_weed_count"])
            or (stage_info["vulnerability"] in ("CRITICAL", "HIGH") and (adjacent_weeds >= self.config["high_risk_crop_proximity_count"] or crop_weed_overlap))
            or (stage_info["vulnerability"] == "CRITICAL" and visible_weeds >= self.config["moderate_risk_weed_count"])
        ):
            risk_category = "HIGH"
            factors = []
            if visible_weeds >= self.config["high_risk_weed_count"]:
                factors.append(f"heavy visible weed count ({visible_weeds})")
            if adjacent_weeds > 0:
                factors.append(f"{adjacent_weeds} weeds adjacent to crop canopy")
            if crop_weed_overlap:
                factors.append("direct bounding-box canopy overlap")
            if stage_info["vulnerability"] in ("CRITICAL", "HIGH"):
                factors.append(f"crop is in sensitive competition phase ({stage_info['cpwc_days']})")

            explanation = (
                f"High crop competition risk observed: {', '.join(factors)}. "
                "Unchecked weed growth during this period can restrict root volume, reduce boll-bearing branches, "
                "and induce nutrient stress."
            )
            priority = "HIGH_PRIORITY"
            requires_field_verification = True

        elif (
            visible_weeds >= self.config["moderate_risk_weed_count"]
            or is_concentrated
            or adjacent_weeds >= 1
        ):
            risk_category = "MODERATE"
            factors = [f"{visible_weeds} weeds detected"]
            if is_concentrated:
                factors.append("clustered in localized patches")
            if adjacent_weeds > 0:
                factors.append(f"{adjacent_weeds} in close proximity to crop")

            explanation = (
                f"Moderate weed pressure observed ({', '.join(factors)}). "
                "Prompt field scouting recommended to plan targeted cultivation or spot intervention."
            )
            priority = "MEDIUM_PRIORITY"
            requires_field_verification = True

        else:
            risk_category = "LOW"
            explanation = (
                f"Low weed pressure observed ({visible_weeds} scattered weed(s)). "
                "Weeds are sparse and not in immediate contact with the crop canopy."
            )
            priority = "ROUTINE_MONITORING"
            requires_field_verification = False

        # Gather supporting evidence
        evidence.append(f"Visible weeds: {visible_weeds}, Visible cotton plants: {visible_cotton}")
        evidence.append(f"Spatial pattern: {dispersion.get('distribution_type', 'UNKNOWN')}")
        if adjacent_weeds > 0:
            evidence.append(f"Weeds within proximity threshold: {adjacent_weeds}")

        return {
            "risk_category": risk_category,
            "risk_explanation": explanation,
            "evidence": evidence,
            "missing_information": missing_info,
            "assessment_priority": priority,
            "requires_field_verification": requires_field_verification,
            "vulnerability_stage": stage_info,
            "scientific_limitation_note": (
                "Risk classification reflects observable frame evidence only and does not predict "
                "yield losses without field trials and certified agronomist calibration."
            ),
        }
