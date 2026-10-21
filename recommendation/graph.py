"""LangGraph Orchestration for AgriSight AI Farmer Advisor.

Implements the 9-node decision workflow:
1. input_validation
2. weed_analysis
3. crop_risk_assessment
4. missing_information_check
5. knowledge_retrieval
6. treatment_assessment
7. safety_validation
8. farmer_explanation
9. final_response

Uses LangGraph StateGraph when available, with a resilient built-in
sequential runner fallback to ensure 100% operational uptime without
requiring external network access.
"""
from typing import Dict, List, Optional, Any, TypedDict
import copy

from analysis.weed_analyzer import WeedAnalyzer
from analysis.spatial_analyzer import SpatialAnalyzer
from analysis.crop_risk_engine import CropRiskEngine
from .knowledge_base import AgriculturalKnowledgeBase
from .recommendation_engine import HybridRecommendationEngine, SafetyValidator
from .groq_service import GroqAdvisorService


class AdvisorState(TypedDict):
    """Shared state dictionary passed across all workflow nodes."""
    raw_boxes: List[Dict[str, Any]]
    image_width: int
    image_height: int
    field_context: Dict[str, Any]
    cumulative_tracked_ids: Optional[Dict[str, set]]
    is_valid_input: bool
    input_errors: List[str]
    weed_analysis: Dict[str, Any]
    spatial_analysis: Dict[str, Any]
    crop_risk: Dict[str, Any]
    has_critical_missing_info: bool
    missing_info_list: List[str]
    knowledge_entries: List[Dict[str, Any]]
    raw_recommendation: Dict[str, Any]
    validated_recommendation: Dict[str, Any]
    farmer_friendly_explanation: str
    final_output: Dict[str, Any]


class RecommendationWorkflow:
    """Orchestrates the 9-node recommendation workflow using LangGraph or internal runner."""

    def __init__(
        self,
        weed_analyzer: Optional[WeedAnalyzer] = None,
        spatial_analyzer: Optional[SpatialAnalyzer] = None,
        crop_risk_engine: Optional[CropRiskEngine] = None,
        knowledge_base: Optional[AgriculturalKnowledgeBase] = None,
        hybrid_engine: Optional[HybridRecommendationEngine] = None,
        groq_service: Optional[GroqAdvisorService] = None,
    ):
        self.weed_analyzer = weed_analyzer or WeedAnalyzer()
        self.spatial_analyzer = spatial_analyzer or SpatialAnalyzer()
        self.crop_risk_engine = crop_risk_engine or CropRiskEngine()
        self.kb = knowledge_base or AgriculturalKnowledgeBase()
        self.hybrid_engine = hybrid_engine or HybridRecommendationEngine(self.kb)
        self.safety_validator = SafetyValidator()
        self.groq_service = groq_service if groq_service is not None else GroqAdvisorService()
        self.graph = self._compile_graph()

    # --- NODE IMPLEMENTATIONS ---

    def node_input_validation(self, state: AdvisorState) -> AdvisorState:
        """Node 1: Validate input bounding boxes and frame geometry."""
        boxes = state.get("raw_boxes", [])
        w = state.get("image_width", 0)
        h = state.get("image_height", 0)
        errors = []

        if not isinstance(boxes, list):
            errors.append("Invalid detection boxes format; expected list.")
        if w <= 0 or h <= 0:
            errors.append(f"Invalid frame dimensions ({w}x{h}).")

        cleaned_boxes = []
        if isinstance(boxes, list):
            for b in boxes:
                if isinstance(b, dict) and "kind" in b:
                    cleaned_boxes.append(b)

        state["raw_boxes"] = cleaned_boxes
        state["is_valid_input"] = len(errors) == 0
        state["input_errors"] = errors
        return state

    def node_weed_analysis(self, state: AdvisorState) -> AdvisorState:
        """Node 2: Run weed analysis (counts, proportions, tracking)."""
        w = state.get("image_width", 1280)
        h = state.get("image_height", 720)
        boxes = state.get("raw_boxes", [])
        ctx = state.get("field_context", {})
        cum_ids = state.get("cumulative_tracked_ids")

        analysis = self.weed_analyzer.analyze(
            boxes=boxes,
            image_width=w,
            image_height=h,
            field_context=ctx,
            cumulative_tracked_ids=cum_ids,
        )
        spatial = self.spatial_analyzer.analyze(
            boxes=boxes,
            image_width=w,
            image_height=h,
            gps_metadata=ctx.get("gps"),
        )

        state["weed_analysis"] = analysis
        state["spatial_analysis"] = spatial
        return state

    def node_crop_risk_assessment(self, state: AdvisorState) -> AdvisorState:
        """Node 3: Assess crop risk from weed presence and context."""
        weed_an = state.get("weed_analysis", {})
        spatial_an = state.get("spatial_analysis", {})
        ctx = state.get("field_context", {})

        risk = self.crop_risk_engine.evaluate_risk(
            weed_analysis=weed_an,
            spatial_analysis=spatial_an,
            field_context=ctx,
        )
        state["crop_risk"] = risk
        return state

    def node_missing_information_check(self, state: AdvisorState) -> AdvisorState:
        """Node 4: Identify whether critical agronomic information is absent."""
        weed_an = state.get("weed_analysis", {})
        risk = state.get("crop_risk", {})

        missing = list(
            set(
                weed_an.get("missing_information", [])
                + risk.get("missing_information", [])
            )
        )
        state["missing_info_list"] = missing
        # Critical if stage unknown AND weeds present in significant numbers
        visible_weeds = weed_an.get("counts", {}).get("visible_weed_count", 0)
        is_critical = visible_weeds >= 3 and any("stage" in m.lower() for m in missing)
        state["has_critical_missing_info"] = is_critical
        return state

    def node_knowledge_retrieval(self, state: AdvisorState) -> AdvisorState:
        """Node 5: Query agricultural knowledge base for verified extension entries."""
        ctx = state.get("field_context", {})
        stage = str(ctx.get("growth_stage", "UNKNOWN"))
        entries = self.kb.retrieve(weed_type="grass", crop_stage=stage, limit=4)
        state["knowledge_entries"] = entries
        return state

    def node_treatment_assessment(self, state: AdvisorState) -> AdvisorState:
        """Node 6: Synthesize raw recommendation and IWM options."""
        rec = self.hybrid_engine.generate_recommendation(
            weed_analysis=state.get("weed_analysis", {}),
            spatial_analysis=state.get("spatial_analysis", {}),
            crop_risk=state.get("crop_risk", {}),
            field_context=state.get("field_context", {}),
        )
        state["raw_recommendation"] = rec
        return state

    def node_safety_validation(self, state: AdvisorState) -> AdvisorState:
        """Node 7: Apply strict safety validation layer."""
        raw_rec = state.get("raw_recommendation", {})
        ctx = state.get("field_context", {})
        validated = self.safety_validator.validate(copy.deepcopy(raw_rec), ctx)
        state["validated_recommendation"] = validated
        return state

    def node_farmer_explanation(self, state: AdvisorState) -> AdvisorState:
        """Node 8: Generate friendly, transparent farmer guidance using Groq or deterministic fallback."""
        rec = state.get("validated_recommendation", {})
        ctx = state.get("field_context", {})
        field_id = ctx.get("field_id", "Field-Plot")
        growth_stage = ctx.get("growth_stage", "UNKNOWN")

        # Attempt AI generation via Groq if available
        if self.groq_service and self.groq_service.is_available:
            try:
                ai_rec = self.groq_service.generate_recommendation(
                    field_id=field_id,
                    growth_stage=growth_stage,
                    weed_analysis=state.get("weed_analysis", {}),
                    spatial_analysis=state.get("spatial_analysis", {}),
                    crop_risk=state.get("crop_risk", {}),
                    knowledge_entries=state.get("knowledge_entries", []),
                    missing_info=state.get("missing_info_list", []),
                    deterministic_fallback=rec,
                )
                # Re-validate with SafetyValidator to guarantee all herbicide/PPE safety constraints
                validated_ai_rec = self.safety_validator.validate(ai_rec, ctx)
                state["validated_recommendation"] = validated_ai_rec
                state["farmer_friendly_explanation"] = validated_ai_rec.get("farmer_summary", "")
                return state
            except Exception:
                # Silently proceed to deterministic template fallback on any Groq failure
                pass

        # Deterministic Rule-Based Guidance Fallback
        weed_an = state.get("weed_analysis", {})
        counts = weed_an.get("counts", {})

        weeds = counts.get("visible_weed_count", 0)
        cotton = counts.get("visible_cotton_count", 0)
        priority = rec.get("action_priority", "ROUTINE_MONITORING")
        reason = rec.get("reason", "")
        follow_up = rec.get("follow_up", "")

        lines = [
            f"🌾 Assessment Overview: {weeds} weed(s) and {cotton} cotton plant(s) in view.",
            f"⚡ Action Priority: {priority.replace('_', ' ')}.",
            f"💡 Agronomic Reason: {reason}",
            f"📅 Follow-up Guidance: {follow_up}",
        ]
        if state.get("missing_info_list"):
            lines.append("❓ Questions for Field Ground-Truth:")
            for item in state["missing_info_list"][:3]:
                lines.append(f"  • {item}")

        rec["ai_provider"] = "Deterministic Fallback"
        state["validated_recommendation"] = rec
        state["farmer_friendly_explanation"] = "\n".join(lines)
        return state

    def node_final_response(self, state: AdvisorState) -> AdvisorState:
        """Node 9: Compile final standardized structured response."""
        final_rec = copy.deepcopy(state.get("validated_recommendation", {}))
        final_rec["farmer_summary"] = state.get("farmer_friendly_explanation", "")
        final_rec["workflow_audit"] = {
            "is_valid_input": state.get("is_valid_input", True),
            "input_errors": state.get("input_errors", []),
            "has_critical_missing_info": state.get("has_critical_missing_info", False),
            "kb_entries_retrieved": len(state.get("knowledge_entries", [])),
            "ai_provider": final_rec.get("ai_provider", "Groq" if (self.groq_service and self.groq_service.is_available) else "Deterministic Fallback"),
            "ai_model": final_rec.get("ai_model", getattr(self.groq_service, "model", "None")),
        }
        state["final_output"] = final_rec
        return state

    # --- LANGGRAPH / FALLBACK COMPILATION ---

    def _compile_graph(self):
        """Build LangGraph StateGraph if installed, else initialize sequential runner."""
        try:
            from langgraph.graph import StateGraph, END

            builder = StateGraph(AdvisorState)

            builder.add_node("input_validation", self.node_input_validation)
            builder.add_node("weed_analysis", self.node_weed_analysis)
            builder.add_node("crop_risk_assessment", self.node_crop_risk_assessment)
            builder.add_node("missing_information_check", self.node_missing_information_check)
            builder.add_node("knowledge_retrieval", self.node_knowledge_retrieval)
            builder.add_node("treatment_assessment", self.node_treatment_assessment)
            builder.add_node("safety_validation", self.node_safety_validation)
            builder.add_node("farmer_explanation", self.node_farmer_explanation)
            builder.add_node("final_response", self.node_final_response)

            builder.set_entry_point("input_validation")
            builder.add_edge("input_validation", "weed_analysis")
            builder.add_edge("weed_analysis", "crop_risk_assessment")
            builder.add_edge("crop_risk_assessment", "missing_information_check")
            builder.add_edge("missing_information_check", "knowledge_retrieval")
            builder.add_edge("knowledge_retrieval", "treatment_assessment")
            builder.add_edge("treatment_assessment", "safety_validation")
            builder.add_edge("safety_validation", "farmer_explanation")
            builder.add_edge("farmer_explanation", "final_response")
            builder.add_edge("final_response", END)

            return builder.compile()
        except Exception:
            # Fallback to pure-Python runner
            return None

    def run(
        self,
        boxes: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        field_context: Optional[Dict[str, Any]] = None,
        cumulative_tracked_ids: Optional[Dict[str, set]] = None,
    ) -> Dict[str, Any]:
        """Execute the complete 9-node advisor workflow."""
        initial_state: AdvisorState = {
            "raw_boxes": boxes,
            "image_width": image_width,
            "image_height": image_height,
            "field_context": field_context or {},
            "cumulative_tracked_ids": cumulative_tracked_ids,
            "is_valid_input": True,
            "input_errors": [],
            "weed_analysis": {},
            "spatial_analysis": {},
            "crop_risk": {},
            "has_critical_missing_info": False,
            "missing_info_list": [],
            "knowledge_entries": [],
            "raw_recommendation": {},
            "validated_recommendation": {},
            "farmer_friendly_explanation": "",
            "final_output": {},
        }

        if self.graph is not None:
            try:
                result = self.graph.invoke(initial_state)
                return result.get("final_output", {})
            except Exception:
                pass  # Use fallback runner if graph invocation hits unexpected error

        # Sequential Fallback Runner
        s = self.node_input_validation(initial_state)
        s = self.node_weed_analysis(s)
        s = self.node_crop_risk_assessment(s)
        s = self.node_missing_information_check(s)
        s = self.node_knowledge_retrieval(s)
        s = self.node_treatment_assessment(s)
        s = self.node_safety_validation(s)
        s = self.node_farmer_explanation(s)
        s = self.node_final_response(s)
        return s.get("final_output", {})


def build_farmer_advisor_graph() -> RecommendationWorkflow:
    """Factory function returning configured RecommendationWorkflow instance."""
    return RecommendationWorkflow()
