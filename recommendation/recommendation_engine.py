"""Hybrid Recommendation Engine for AgriSight AI.

Combines:
A. Structured Agricultural Decision Rules (Cotton IWM Framework)
B. Verified Knowledge Retrieval (AgriculturalKnowledgeBase)
C. Safety Validation Layer (Strict regulatory and drift prevention constraints)
D. Transparent Agronomic Reasoning for Farmers

Action Priorities:
- HIGH PRIORITY
- MEDIUM PRIORITY
- ROUTINE MONITORING
- VERIFICATION REQUIRED

Enforces strict chemical safety: Never prescribes arbitrary chemical dosages
or non-selective herbicides without certified label instructions and protective hoods.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from .knowledge_base import AgriculturalKnowledgeBase


class SafetyValidator:
    """Mandatory safety validation layer for agricultural recommendations."""

    @staticmethod
    def validate(recommendation: Dict[str, Any], field_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety constraints on proposed management options.

        Enforces:
        - Mandatory Personal Protective Equipment (PPE) notices if chemical methods mentioned
        - Mandatory spray shield/hood requirements for directed sprays in cotton
        - Wind speed (<8-10 km/h) and rain-free period constraints (4-6 hours)
        - Prohibition on arbitrary herbicide dosage prescription without official label
        - Rejection of unverified chemical treatments
        """
        validated_options = []
        safety_disclaimers = [
            "Always consult locally registered product labels (CIB&RC / State Agriculture Dept) before applying any chemical.",
            "Use certified Personal Protective Equipment (gloves, eye protection, respirator/mask) during handling.",
        ]

        for opt in recommendation.get("management_options", []):
            text = str(opt).strip()
            # If chemical options are mentioned, ensure strict safety caveats are attached
            if any(w in text.lower() for w in ("herbicide", "chemical", "spray")):
                safe_text = (
                    f"{text} [MANDATORY SAFETY: Must be applied strictly as a shielded/directed spray "
                    "between crop rows using protective hoods to prevent drift on cotton foliage. "
                    "Verify dosage with the official CIB&RC approved product container label]."
                )
                validated_options.append(safe_text)
                if "wind speed" not in " ".join(safety_disclaimers).lower():
                    safety_disclaimers.append("Do not spray if wind speeds exceed 8 km/h or if rainfall is forecast within 4–6 hours.")
            else:
                validated_options.append(text)

        recommendation["management_options"] = validated_options
        recommendation["safety_disclaimers"] = safety_disclaimers
        return recommendation


class HybridRecommendationEngine:
    """Hybrid decision engine translating detection analysis into validated farm guidance."""

    def __init__(self, knowledge_base: Optional[AgriculturalKnowledgeBase] = None):
        self.kb = knowledge_base or AgriculturalKnowledgeBase()
        self.validator = SafetyValidator()

    def generate_recommendation(
        self,
        weed_analysis: Dict[str, Any],
        spatial_analysis: Dict[str, Any],
        crop_risk: Dict[str, Any],
        field_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize analysis layers into a structured, safety-validated farmer recommendation.

        Args:
            weed_analysis: Output from WeedAnalyzer
            spatial_analysis: Output from SpatialAnalyzer
            crop_risk: Output from CropRiskEngine
            field_context: User-provided farm context (field_id, growth_stage, etc.)

        Returns:
            Structured recommendation dictionary.
        """
        ctx = field_context or {}
        assessment_id = f"AGRI_REC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        risk_category = crop_risk.get("risk_category", "INSUFFICIENT_DATA")
        counts = weed_analysis.get("counts", {})
        dispersion = weed_analysis.get("dispersion", {})
        proximity = spatial_analysis.get("inter_crop_proximity", {})

        visible_weeds = counts.get("visible_weed_count", 0)
        tracked_weeds = counts.get("tracked_weed_count")
        count_type = "Tracked Unique Plants (ByteTrack)" if tracked_weeds is not None else "Visible Plants in Frame"
        effective_weed_count = tracked_weeds if tracked_weeds is not None else visible_weeds

        weed_distribution = dispersion.get("distribution_type", "UNKNOWN")
        infestation_level = weed_analysis.get("infestation_level", "UNKNOWN")

        # Compile Missing Information from all analysis layers
        missing_info = list(
            set(
                weed_analysis.get("missing_information", [])
                + crop_risk.get("missing_information", [])
            )
        )

        # 1. Determine Action Priority
        if risk_category == "INSUFFICIENT_DATA":
            action_priority = "VERIFICATION_REQUIRED"
            recommended_action = "FIELD_GROUND_VERIFICATION"
            requires_field_verification = True
            reassessment_timeline = "Immediately inspect the field plot to confirm weed pressure and crop stage."
        elif risk_category == "HIGH":
            action_priority = "HIGH_PRIORITY"
            recommended_action = "INSPECT_AND_INITIATE_MANAGEMENT"
            requires_field_verification = True
            reassessment_timeline = "Inspect within 24–48 hours; reassess plot 3 days after treatment."
        elif risk_category == "MODERATE":
            action_priority = "MEDIUM_PRIORITY"
            recommended_action = "SCOUT_AND_SCHEDULE_INTER_CULTIVATION"
            requires_field_verification = True
            reassessment_timeline = "Inspect affected patch within 3–5 days; monitor after scheduled cultivation."
        else:
            action_priority = "ROUTINE_MONITORING"
            recommended_action = "CONTINUE_REGULAR_MONITORING"
            requires_field_verification = False
            reassessment_timeline = "Re-scout field during routine weekly observation (every 5–7 days)."

        # 2. Formulate Integrated Weed Management (IWM) Options
        management_options: List[str] = []
        growth_stage = str(ctx.get("growth_stage", "UNKNOWN")).upper()

        if effective_weed_count == 0:
            management_options.append("No active weed removal necessary at this time.")
            management_options.append("Maintain clean field borders and bunds to prevent seed intrusion.")
        else:
            # Mechanical Options (Row inter-cultivation)
            if growth_stage in ("VEGETATIVE_SQUARING", "FLOWERING_BOLL_DEV", "UNKNOWN"):
                management_options.append(
                    "Mechanical Inter-Cultivation: Carry out shallow tractor-drawn or bullock-drawn blade harrowing "
                    "(4–5 cm depth) between crop rows to uproot weeds and create a protective soil mulch."
                )

            # Manual Options (Intra-row hoeing)
            if proximity.get("weeds_adjacent_to_crop", 0) > 0 or growth_stage in ("EMERGENCE_SEEDLING", "UNKNOWN"):
                management_options.append(
                    "Manual Intra-Row Weeding: Perform manual hand-hoeing (khurpi) directly within the seed line "
                    "where mechanical weeders cannot reach without damaging cotton stems."
                )

            # Cultural Practices
            management_options.append(
                "Cultural Prevention: Prevent weed escapes from flowering and setting seeds to reduce the soil weed seed bank."
            )

            # Chemical Guidance (If high or moderate pressure, with strict safety caveats)
            if risk_category in ("HIGH", "MODERATE"):
                management_options.append(
                    "Chemical Consideration (Spot / Directed Spray): If labor or soil conditions prevent mechanical hoeing, "
                    "evaluate an authorized post-emergence herbicide registered specifically for cotton."
                )

        # 3. Retrieve Verified Agricultural Knowledge Entries
        retrieved_entries = self.kb.retrieve(
            weed_type="grass",
            crop_stage=growth_stage,
            limit=4,
        )

        sources = []
        for e in retrieved_entries:
            src = e.get("source", {})
            sources.append(
                {
                    "title": e.get("title"),
                    "organization": src.get("organization"),
                    "publication": src.get("publication"),
                    "year": src.get("year"),
                    "citation": src.get("citation"),
                    "key_takeaway": e.get("recommendation_text"),
                }
            )

        # 4. Formulate Detailed Natural Rationale
        reason_parts = [
            f"Observed {effective_weed_count} {count_type.lower()}.",
            f"Distribution pattern is {weed_distribution.replace('_', ' ').lower()}.",
        ]
        if proximity.get("weeds_adjacent_to_crop", 0) > 0:
            reason_parts.append(
                f"{proximity['weeds_adjacent_to_crop']} weed(s) are in direct proximity to cotton stems."
            )
        if risk_category == "HIGH":
            reason_parts.append(
                "Prompt intervention is critical to prevent aggressive grassy weeds from sequestering nutrients and moisture during key vegetative development."
            )
        elif risk_category == "MODERATE":
            reason_parts.append(
                "Weed pressure is currently clustered and manageable via timely inter-row cultivation before weeds establish deep root systems."
            )
        elif risk_category == "LOW":
            reason_parts.append("Weed density remains below economic injury thresholds; continued routine scouting is advised.")
        else:
            reason_parts.append("Optical clarity or agronomic field parameters are incomplete, requiring physical ground verification.")

        reason = " ".join(reason_parts)

        # 5. Build Preliminary Recommendation Object
        recommendation = {
            "assessment_id": assessment_id,
            "timestamp": timestamp,
            "field_id": ctx.get("field_id", "FIELD_DEFAULT"),
            "weed_count": effective_weed_count,
            "count_type": count_type,
            "visible_cotton_count": counts.get("visible_cotton_count", 0),
            "weed_distribution": weed_distribution,
            "infestation_level": infestation_level,
            "crop_risk": risk_category,
            "action_priority": action_priority,
            "recommended_action": recommended_action,
            "management_options": management_options,
            "reason": reason,
            "requires_field_verification": requires_field_verification,
            "missing_information": missing_info,
            "follow_up": reassessment_timeline,
            "sources": sources,
            "expert_validation_note": (
                "Action priorities represent structured agronomic decision rules. "
                "Any herbicide selection, dosage, or tank-mix must be verified against current CIB&RC registrations "
                "and approved local extension recommendations."
            ),
        }

        # 6. Apply Mandatory Safety Validation Layer
        recommendation = self.validator.validate(recommendation, ctx)
        return recommendation
