"""Groq API Integration Service for AgriSight AI Recommendation Engine.

Provides an isolated, production-grade agronomic advisory service powered by Groq's
high-speed LPU inference engine, grounded strictly in validated computer vision detections
and certified extension guidelines (ICAR, TNAU, PAU, CIB&RC).

Key Safety & Integrity Guardrails:
1. Strict Determinism: Groq CANNOT alter the deterministic crop-risk level (LOW, MODERATE, HIGH, INSUFFICIENT_DATA).
2. Grounded Agronomy: The model is constrained to the verified agricultural knowledge base.
3. No Dosage Fabrication: The model is explicitly forbidden from inventing chemical dosages or unregistered products.
4. Robust Fallback: If API key is missing, network fails, or output is invalid, falls back seamlessly to deterministic engine.
5. In-Memory Deduplication: Hashes input state to avoid duplicate API round-trips for identical frames.
"""
import os
import json
import re
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Load local environment variables if available
try:
    from dotenv import load_dotenv
    # Look for .env in current directory and parent directory
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger("AgriSight.GroqService")

# Allowed deterministic risk levels
VALID_RISK_LEVELS = {"LOW", "MODERATE", "HIGH", "INSUFFICIENT_DATA"}


class GroqServiceError(Exception):
    """Base exception for Groq service issues."""
    pass


class GroqAdvisorService:
    """Enterprise client for Groq-powered precision agriculture recommendations."""

    DEFAULT_FALLBACK_MODELS = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 12.0,
        cache_ttl_seconds: float = 30.0,
    ):
        """Initialize Groq client securely from environment or explicit key."""
        if api_key is None:
            self._api_key = os.getenv("GROQ_API_KEY", "").strip()
        else:
            self._api_key = api_key.strip()

        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        self.timeout = timeout_seconds
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        self._client = None
        self._is_configured = False
        self._last_error = ""

        self._init_client()

    def _init_client(self):
        """Safely instantiate Groq client without throwing unhandled exceptions."""
        if not self._api_key:
            self._is_configured = False
            self._last_error = "GROQ_API_KEY is not configured in backend environment."
            return

        try:
            from groq import Groq
            self._client = Groq(api_key=self._api_key, timeout=self.timeout)
            self._is_configured = True
            self._last_error = ""
        except ImportError:
            self._is_configured = False
            self._last_error = "The 'groq' Python package is not installed."
        except Exception as exc:
            self._is_configured = False
            self._last_error = f"Failed to initialize Groq client: {exc}"

    @property
    def is_available(self) -> bool:
        """Return True if client is initialized with an API key."""
        return self._is_configured and self._client is not None

    def get_status(self) -> Dict[str, Any]:
        """Return operational diagnostic status without exposing secret key."""
        return {
            "configured": self._is_configured,
            "has_api_key": bool(self._api_key),
            "model": self.model,
            "timeout_seconds": self.timeout,
            "last_error": self._last_error,
        }

    def _compute_state_hash(self, payload: Dict[str, Any]) -> str:
        """Compute deterministic SHA-256 hash of structured inputs for deduplication."""
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def build_structured_prompt_context(
        self,
        field_id: str,
        growth_stage: str,
        weed_analysis: Dict[str, Any],
        spatial_analysis: Dict[str, Any],
        crop_risk: Dict[str, Any],
        knowledge_entries: List[Dict[str, Any]],
        missing_info: List[str],
    ) -> Dict[str, Any]:
        """Filter and prepare actual structured inputs without inventing data."""
        counts = weed_analysis.get("counts", {})
        dispersion = weed_analysis.get("dispersion", {})
        proximity = spatial_analysis.get("inter_crop_proximity", {})

        # Extract only actually observed data
        actual_context: Dict[str, Any] = {
            "field_id": field_id or "Field-Plot",
            "cotton_growth_stage": growth_stage if growth_stage and growth_stage != "UNKNOWN" else "Not Specified",
            "visible_weed_count": counts.get("visible_weed_count", 0),
            "visible_cotton_count": counts.get("visible_cotton_count", 0),
            "infestation_level": weed_analysis.get("infestation_level", "UNKNOWN"),
            "distribution_type": dispersion.get("distribution_type", "UNKNOWN"),
            "existing_crop_risk_category": crop_risk.get("risk_category", "INSUFFICIENT_DATA"),
            "crop_risk_score": crop_risk.get("risk_score", 0.0),
        }

        # Include tracked counts only if ByteTrack tracking was active
        if counts.get("tracked_weed_count") is not None:
            actual_context["unique_tracked_weed_count"] = counts["tracked_weed_count"]
        if counts.get("tracked_cotton_count") is not None:
            actual_context["unique_tracked_cotton_count"] = counts["tracked_cotton_count"]

        # Include weed-to-crop ratio if available
        if "weed_to_crop_ratio" in counts:
            actual_context["weed_to_crop_ratio"] = counts["weed_to_crop_ratio"]

        # Spatial clusters and proximity
        if dispersion.get("cluster_count") is not None:
            actual_context["weed_clusters_identified"] = dispersion["cluster_count"]
        if proximity.get("weeds_adjacent_to_crop") is not None:
            actual_context["weeds_in_direct_cotton_contact"] = proximity["weeds_adjacent_to_crop"]

        # Missing information warnings
        if missing_info:
            actual_context["missing_information_warnings"] = missing_info

        # Compile verified knowledge base references
        verified_references = []
        for entry in knowledge_entries:
            src = entry.get("source", {})
            verified_references.append({
                "title": entry.get("title", "Agricultural Recommendation"),
                "organization": src.get("organization", "Agricultural University / ICAR"),
                "publication": src.get("publication", "Package of Practices"),
                "year": src.get("year", "Recent Edition"),
                "citation": src.get("citation", ""),
                "recommendation_text": entry.get("recommendation_text", ""),
                "management_type": entry.get("management_type", "IWM"),
            })

        actual_context["verified_agricultural_references"] = verified_references
        return actual_context

    def generate_recommendation(
        self,
        field_id: str,
        growth_stage: str,
        weed_analysis: Dict[str, Any],
        spatial_analysis: Dict[str, Any],
        crop_risk: Dict[str, Any],
        knowledge_entries: List[Dict[str, Any]],
        missing_info: List[str],
        deterministic_fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured recommendation via Groq or fallback safely.

        Args:
            field_id: Identifier of field/plot.
            growth_stage: Cotton growth stage.
            weed_analysis: Output from WeedAnalyzer.
            spatial_analysis: Output from SpatialAnalyzer.
            crop_risk: Output from CropRiskEngine.
            knowledge_entries: Retrieved ICAR/TNAU/PAU citations.
            missing_info: List of identified missing parameters.
            deterministic_fallback: Precomputed rule-based recommendation.

        Returns:
            Structured recommendation dictionary matching frontend format.
        """
        deterministic_risk = crop_risk.get("risk_category", "INSUFFICIENT_DATA")
        if deterministic_risk not in VALID_RISK_LEVELS:
            deterministic_risk = "INSUFFICIENT_DATA"

        if not self.is_available:
            # Clean fallback to deterministic engine
            fb = dict(deterministic_fallback)
            fb["ai_provider"] = "Deterministic Fallback"
            fb["ai_status"] = "API_KEY_NOT_CONFIGURED" if not self._api_key else "CLIENT_UNAVAILABLE"
            return fb

        # Build clean structured context
        context_payload = self.build_structured_prompt_context(
            field_id=field_id,
            growth_stage=growth_stage,
            weed_analysis=weed_analysis,
            spatial_analysis=spatial_analysis,
            crop_risk=crop_risk,
            knowledge_entries=knowledge_entries,
            missing_info=missing_info,
        )

        # Check deduplication cache
        cache_key = self._compute_state_hash(context_payload)
        now = time.monotonic()
        if cache_key in self._cache:
            ts, cached_response = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                cached_copy = dict(cached_response)
                cached_copy["cached"] = True
                return cached_copy

        # Generate via Groq
        try:
            groq_raw = self._call_groq_api(context_payload, deterministic_risk)
            validated = self._validate_and_sanitize_response(
                groq_response=groq_raw,
                deterministic_risk=deterministic_risk,
                field_id=field_id,
                context_payload=context_payload,
                fallback_rec=deterministic_fallback,
            )

            # Mark provider metadata
            validated["ai_provider"] = "Groq"
            validated["ai_model"] = self.model
            validated["cached"] = False

            # Cache the validated response
            self._cache[cache_key] = (now, validated)
            if len(self._cache) > 40:
                oldest = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest]

            return validated

        except Exception as exc:
            logger.warning(f"Groq API call/validation failed, falling back: {exc}")
            self._last_error = str(exc)
            fb = dict(deterministic_fallback)
            fb["ai_provider"] = "Deterministic Fallback"
            fb["ai_status"] = "FALLBACK_ON_ERROR"
            fb["ai_fallback_reason"] = str(exc)[:200]
            return fb

    def _call_groq_api(self, context_payload: Dict[str, Any], deterministic_risk: str) -> Dict[str, Any]:
        """Invoke Groq chat completion endpoint with strict system instruction."""
        from groq import NotFoundError, RateLimitError, APIConnectionError, AuthenticationError

        system_prompt = (
            "You are AgriSight AI's Senior Agricultural Advisor and Agronomist specializing in Indian cotton cultivation.\n"
            "Your objective is to provide actionable, farmer-friendly, grounded Integrated Weed Management (IWM) advice.\n\n"
            "STRICT RULES & CONSTRAINTS:\n"
            "1. Output MUST be a valid JSON object matching the requested schema.\n"
            f"2. You MUST set 'risk_level' EXACTLY to '{deterministic_risk}'. Do NOT override or change this value.\n"
            "3. Ground all agronomic reasoning strictly in the provided 'verified_agricultural_references' (ICAR-CICR, TNAU, PAU, CIB&RC).\n"
            "4. NEVER invent or fabricate chemical dosages, unregistered pesticide names, or chemical trade names.\n"
            "5. If chemical control is referenced, you MUST include the regulatory constraint: 'Directed/shielded inter-row spray only; verify dosage with official CIB&RC label; PPE mandatory; do not spray if wind > 8 km/h or rain expected within 4-6 hours.'\n"
            "6. DO NOT invent weed species (the camera detector identifies generic grass or weeds; do not pretend to know species unless verified).\n"
            "7. DO NOT invent GPS coordinates, exact physical kilograms of weed biomass, or yield-loss percentages.\n"
            "8. Keep the farmer summary transparent, respectful, encouraging, and clear for practical farm laborers and smallholders.\n"
            "9. Return ONLY the JSON object. Do not include introductory text, conversational pleasantries, or <think> reasoning blocks."
        )

        user_content = (
            "Analyze the following validated field detection results and verified agricultural sources:\n\n"
            f"{json.dumps(context_payload, indent=2)}\n\n"
            "Generate your recommendation in strictly following JSON format:\n"
            "{\n"
            f'  "field_id": "{context_payload.get("field_id")}",\n'
            '  "summary": "1-2 sentence farmer-friendly executive summary",\n'
            f'  "risk_level": "{deterministic_risk}",\n'
            '  "observations": ["Key observation 1 from detections", "Key observation 2"],\n'
            '  "recommended_action": "Primary actionable next step (e.g., Mechanical Inter-Cultivation or Field Scouting)",\n'
            '  "reasoning": "Clear agronomic explanation grounded in crop stage, weed competition, and references",\n'
            '  "management_options": [\n'
            '    {\n'
            '      "title": "Method name (e.g. Shallow Inter-Row Cultivation)",\n'
            '      "description": "Specific practical instructions for farmer",\n'
            '      "source": "Citation from provided verified references"\n'
            '    }\n'
            '  ],\n'
            '  "missing_information": ["List missing parameters if any, else empty list"],\n'
            '  "safety_warnings": ["Safety precautions including PPE and wind/drift limits"],\n'
            '  "sources": [\n'
            '    {\n'
            '      "title": "Reference title",\n'
            '      "organization": "ICAR-CICR / TNAU / PAU / CIB&RC",\n'
            '      "publication": "Publication name",\n'
            '      "year": "Year",\n'
            '      "citation": "Full citation string",\n'
            '      "key_takeaway": "Key advice summary"\n'
            '    }\n'
            '  ],\n'
            '  "limitations": ["Computer vision confidence, camera coverage scope, need for physical ground check"]\n'
            "}"
        )

        models_to_try = [self.model] + [m for m in self.DEFAULT_FALLBACK_MODELS if m != self.model]

        last_exc = None
        for current_model in models_to_try:
            try:
                completion = self._client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                self.model = current_model
                content = completion.choices[0].message.content or ""
                # Strip think blocks if present
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return json.loads(content)

            except NotFoundError as nf_err:
                # If model is not found on account, try next available model
                logger.info(f"Model '{current_model}' not found on Groq, trying fallback: {nf_err}")
                last_exc = nf_err
                continue
            except (RateLimitError, AuthenticationError, APIConnectionError) as provider_err:
                raise provider_err
            except json.JSONDecodeError as json_err:
                raise GroqServiceError(f"Groq returned malformed JSON: {json_err}")
            except Exception as e:
                last_exc = e
                break

        raise GroqServiceError(f"All Groq model attempts failed. Last error: {last_exc}")

    def _validate_and_sanitize_response(
        self,
        groq_response: Dict[str, Any],
        deterministic_risk: str,
        field_id: str,
        context_payload: Dict[str, Any],
        fallback_rec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate structure, types, safety restrictions, and merge with existing frontend schema."""
        if not isinstance(groq_response, dict):
            raise GroqServiceError("Groq output is not a dictionary.")

        # 1. Enforce strict deterministic risk level (LLM cannot override risk)
        groq_risk = str(groq_response.get("risk_level", "")).upper()
        if groq_risk != deterministic_risk:
            logger.info(f"Correcting Groq risk level '{groq_risk}' to deterministic '{deterministic_risk}'.")
            groq_response["risk_level"] = deterministic_risk

        # 2. Extract and sanitize fields
        summary = str(groq_response.get("summary", "")).strip() or fallback_rec.get("reason", "")
        recommended_action = str(groq_response.get("recommended_action", "")).strip() or fallback_rec.get("recommended_action", "")
        reasoning = str(groq_response.get("reasoning", "")).strip() or fallback_rec.get("reason", "")

        raw_observations = groq_response.get("observations", [])
        if not isinstance(raw_observations, list):
            raw_observations = []
        observations = [str(o).strip() for o in raw_observations if str(o).strip()]

        # 3. Format management options
        raw_options = groq_response.get("management_options", [])
        formatted_options: List[str] = []
        structured_options: List[Dict[str, str]] = []

        if isinstance(raw_options, list):
            for opt in raw_options:
                if isinstance(opt, dict):
                    t = str(opt.get("title", "")).strip()
                    d = str(opt.get("description", "")).strip()
                    s = str(opt.get("source", "")).strip()
                    full_str = f"{t}: {d}" if t else d
                    if s:
                        full_str += f" (Ref: {s})"
                    if full_str:
                        formatted_options.append(full_str)
                        structured_options.append({"title": t or "Management Option", "description": d, "source": s})
                elif isinstance(opt, str) and opt.strip():
                    formatted_options.append(opt.strip())
                    structured_options.append({"title": "Management Option", "description": opt.strip(), "source": ""})

        if not formatted_options:
            formatted_options = fallback_rec.get("management_options", [])

        # 4. Sources validation
        raw_sources = groq_response.get("sources", [])
        valid_sources = []
        if isinstance(raw_sources, list) and len(raw_sources) > 0:
            for s in raw_sources:
                if isinstance(s, dict) and (s.get("title") or s.get("organization")):
                    valid_sources.append({
                        "title": str(s.get("title", "Agricultural Reference")),
                        "organization": str(s.get("organization", "Agricultural Institution")),
                        "publication": str(s.get("publication", "Agronomy Guide")),
                        "year": str(s.get("year", "2024")),
                        "citation": str(s.get("citation", "")),
                        "key_takeaway": str(s.get("key_takeaway", "")),
                    })
        if not valid_sources:
            valid_sources = fallback_rec.get("sources", [])

        # 5. Missing information and limitations
        missing_info = groq_response.get("missing_information")
        if not isinstance(missing_info, list) or not missing_info:
            missing_info = context_payload.get("missing_information_warnings", fallback_rec.get("missing_information", []))

        safety_warnings = groq_response.get("safety_warnings")
        if not isinstance(safety_warnings, list) or not safety_warnings:
            safety_warnings = fallback_rec.get("safety_disclaimers", [])

        limitations = groq_response.get("limitations")
        if not isinstance(limitations, list) or not limitations:
            limitations = [
                "Computer vision detection is based on current optical camera angle and lighting conditions.",
                "Verify weed growth and soil moisture by direct field scouting before applying mechanical or chemical measures.",
            ]

        # 6. Map to both the requested 5. Structured Response Format and backward-compatible fields
        result = dict(fallback_rec)
        result.update({
            "field_id": field_id or context_payload.get("field_id", "Plot-A1"),
            "summary": summary,
            "risk_level": deterministic_risk,
            "crop_risk": deterministic_risk,
            "observations": observations,
            "recommended_action": recommended_action,
            "reasoning": reasoning,
            "reason": reasoning,  # Keep existing frontend compatibility
            "management_options": formatted_options,
            "management_options_structured": structured_options,
            "missing_information": missing_info,
            "safety_warnings": safety_warnings,
            "safety_disclaimers": safety_warnings,
            "sources": valid_sources,
            "limitations": limitations,
            "farmer_summary": f"🌾 {summary}\n⚡ Action: {recommended_action}\n💡 Rationale: {reasoning}",
        })

        return result
