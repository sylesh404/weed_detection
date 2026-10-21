"""AI Farmer Advisor for AgriSight AI.

Provides the primary interface for generating farmer recommendations,
answering agronomic questions, and caching assessments to prevent
unnecessary duplicate computations.
"""
from typing import Dict, List, Optional, Any
import hashlib
import json
import time

from .graph import RecommendationWorkflow, build_farmer_advisor_graph
from .knowledge_base import AgriculturalKnowledgeBase
from .groq_service import GroqAdvisorService


class FarmerAdvisor:
    """High-level advisory agent integrating workflow execution, caching,
    and farmer-facing agronomic Q&A.
    """

    def __init__(
        self,
        workflow: Optional[RecommendationWorkflow] = None,
        knowledge_base: Optional[AgriculturalKnowledgeBase] = None,
        groq_service: Optional[GroqAdvisorService] = None,
        cache_ttl_secs: int = 15,
    ):
        self.kb = knowledge_base or AgriculturalKnowledgeBase()
        self.groq_service = groq_service or (workflow.groq_service if workflow and hasattr(workflow, "groq_service") else GroqAdvisorService())
        self.workflow = workflow or build_farmer_advisor_graph()
        if hasattr(self.workflow, "groq_service") and self.groq_service:
            self.workflow.groq_service = self.groq_service
        self.cache_ttl_secs = cache_ttl_secs
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _compute_cache_key(
        self,
        boxes: List[Dict[str, Any]],
        field_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create deterministic hash signature from boxes and field context."""
        box_sigs = [
            f"{b.get('kind')}:{round(b.get('conf', 0), 2)}:{b.get('track_id')}"
            for b in sorted(boxes, key=lambda x: (x.get("x1", 0), x.get("y1", 0)))
        ]
        ctx_sig = json.dumps(field_context or {}, sort_keys=True)
        raw = f"{'|'.join(box_sigs)}___{ctx_sig}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def advise(
        self,
        boxes: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        field_context: Optional[Dict[str, Any]] = None,
        cumulative_tracked_ids: Optional[Dict[str, set]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Generate a validated agricultural recommendation for the given detections.

        Utilizes signature caching to avoid repeated evaluations on identical frames.
        """
        cache_key = self._compute_cache_key(boxes, field_context)
        now = time.monotonic()

        if not force_refresh and cache_key in self._cache:
            entry = self._cache[cache_key]
            if now - entry["timestamp"] < self.cache_ttl_secs:
                cached_rec = copy_dict(entry["data"])
                cached_rec["cached"] = True
                return cached_rec

        # Execute full workflow
        rec = self.workflow.run(
            boxes=boxes,
            image_width=image_width,
            image_height=image_height,
            field_context=field_context,
            cumulative_tracked_ids=cumulative_tracked_ids,
        )
        rec["cached"] = False

        # Store in cache
        self._cache[cache_key] = {"timestamp": now, "data": rec}
        # Maintain cache size
        if len(self._cache) > 50:
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

        return rec

    def answer_question(self, question: str, current_rec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer farmer follow-up questions strictly within supported agricultural guidance."""
        q = question.lower()
        matched_entries = []

        if any(w in q for w in ("spray", "herbicide", "chemical", "chemical control")):
            matched_entries = self.kb.retrieve(management_types=["CHEMICAL_RESTRICTION", "APPLICATION_TIMING"])
            headline = "Chemical Stewardship & Safety Requirements in Cotton"
            answer = (
                "Cotton foliage is extremely vulnerable to non-selective herbicides. Any post-emergence spray "
                "must be strictly directed between crop rows using protective spray hoods or shields. "
                "Do not spray if wind exceeds 8 km/h or if rainfall is forecast within 4 to 6 hours. "
                "Always verify chemical selection and dosage with the official CIB&RC approved product container label."
            )
        elif any(w in q for w in ("cultivator", "mechanical", "hoeing", "plow", "weeder")):
            matched_entries = self.kb.retrieve(management_types=["MECHANICAL"])
            headline = "Mechanical Weeding Best Practices"
            answer = (
                "Inter-row cultivation (using tractor-drawn cultivators or blade harrows) should be shallow (4–5 cm depth) "
                "to prevent severing lateral cotton feeder roots. Conduct mechanical hoeing between 20 and 45 Days After Sowing. "
                "Avoid operating machinery in wet or waterlogged soil."
            )
        elif any(w in q for w in ("cpwc", "critical period", "when", "timing", "days")):
            matched_entries = self.kb.retrieve(management_types=["CULTURAL_TIMING"])
            headline = "Critical Period of Weed Competition (CPWC)"
            answer = (
                "The critical period of weed competition in cotton occurs during the first 15 to 60 Days After Sowing (DAS). "
                "Weed interference during this window causes irreversible stunting of boll-bearing branches. "
                "Keeping the field weed-free during these initial two months protects maximum yield potential."
            )
        else:
            matched_entries = self.kb.retrieve(limit=3)
            headline = "General Cotton Integrated Weed Management (IWM)"
            answer = (
                "Effective weed management in cotton combines early mechanical inter-row cultivation (20-45 DAS), "
                "intra-row manual hand weeding around seedling bases, and strict field sanitation. "
                "For customized advice, specify your crop growth stage (Days After Sowing) and whether weeds are clustered or scattered."
            )

        citations = []
        for e in matched_entries:
            s = e.get("source", {})
            citations.append(f"{s.get('organization')} — {s.get('publication')} ({s.get('year')})")

        # Attempt Groq for dynamic, context-aware answering grounded in agricultural references
        if self.groq_service and self.groq_service.is_available:
            try:
                system_prompt = (
                    "You are AgriSight AI's Senior Agricultural Advisor answering a cotton farmer's question.\n"
                    "Ground your answer strictly in these verified agricultural extension references:\n"
                    + "\n".join([f"- {c}" for c in citations])
                    + "\n\nCRITICAL CONSTRAINTS:\n"
                    "1. NEVER invent chemical dosages or unregistered chemicals.\n"
                    "2. If chemical weed control is mentioned, always enforce directed/shielded spray, PPE, and no spraying in wind > 8 km/h or rain within 4-6h.\n"
                    "3. Respond in valid JSON with keys 'headline' (short title) and 'answer' (2-4 clear, helpful sentences for a farmer)."
                )
                user_msg = f"Farmer Question: {question}\nCurrent Field Context: {json.dumps(current_rec or {})}"
                client = self.groq_service._client
                if client is not None:
                    chat = client.chat.completions.create(
                        model=self.groq_service.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                    )
                    parsed = json.loads(chat.choices[0].message.content or "{}")
                    if parsed.get("answer"):
                        return {
                            "question": question,
                            "headline": parsed.get("headline", headline),
                            "answer": parsed.get("answer", answer),
                            "supporting_citations": citations,
                            "grounded": True,
                            "ai_provider": "Groq",
                            "ai_model": self.groq_service.model,
                        }
            except Exception:
                pass

        return {
            "question": question,
            "headline": headline,
            "answer": answer,
            "supporting_citations": citations,
            "grounded": True,
            "ai_provider": "Deterministic Fallback",
        }


def copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to shallow/deep copy dict safely."""
    try:
        return json.loads(json.dumps(d))
    except Exception:
        return dict(d)
