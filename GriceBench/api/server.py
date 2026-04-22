"""
api/server.py
==============
GriceBench Production API Server
---------------------------------
Endpoints:
  POST /detect    — Multi-label maxim violation detection
  POST /repair    — Single-maxim response repair
  POST /generate  — Cooperative response generation (DPO model)
  POST /pipeline  — Full detect → generate → repair → re-detect pipeline
  GET  /health    — Health check with model load status
  GET  /metrics   — Prometheus metrics endpoint

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs        (Swagger UI)
    http://localhost:8000/redoc       (ReDoc)
    http://localhost:8000/openapi.json

Requirements:
    fastapi uvicorn[standard] prometheus-client pydantic>=2.0
    transformers peft torch sentencepiece

Author: GriceBench Research
Version: 1.0 — March 2026
"""

import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("gricebench.api")

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "gricebench_requests_total",
    "Total API requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "gricebench_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
VIOLATION_COUNTER = Counter(
    "gricebench_violations_total",
    "Total violations detected per maxim",
    ["maxim"]
)
COOPERATIVE_RATE_GAUGE = Gauge(
    "gricebench_cooperative_rate",
    "Rolling cooperative rate over last 100 /detect requests"
)
REPAIR_FALLBACK_COUNTER = Counter(
    "gricebench_repair_fallbacks_total",
    "Number of repair calls that fell back to original response"
)

# ── Global Request History (for rolling cooperative rate) ─────────────────────
_recent_cooperative: list[int] = []  # 0/1 for last 100 /detect calls
_ROLLING_WINDOW = 100


def _update_cooperative_rate(is_cooperative: bool):
    _recent_cooperative.append(int(is_cooperative))
    if len(_recent_cooperative) > _ROLLING_WINDOW:
        _recent_cooperative.pop(0)
    if _recent_cooperative:
        rate = sum(_recent_cooperative) / len(_recent_cooperative)
        COOPERATIVE_RATE_GAUGE.set(rate)


# ── Pydantic Request/Response Models ─────────────────────────────────────────
class DetectRequest(BaseModel):
    context: str = Field(
        ..., min_length=1, max_length=2000,
        description="Conversation history (up to ~2000 chars)"
    )
    response: str = Field(
        ..., min_length=1, max_length=1000,
        description="Response to evaluate for Gricean violations"
    )
    evidence: Optional[str] = Field(
        None, max_length=1000,
        description="Knowledge evidence snippet (used for Quality violation detection)"
    )

    model_config = {"json_schema_extra": {"example": {
        "context": "Do you enjoy watching movies?",
        "response": "I do enjoy movies! The cinema industry employs millions globally, encompassing production, distribution, exhibition, and ancillary sectors that together generate hundreds of billions of dollars annually...",
        "evidence": None
    }}}


class DetectResponse(BaseModel):
    violations: dict[str, bool]          # {"quantity": True, "quality": False, ...}
    probabilities: dict[str, float]      # {"quantity": 0.97, ...}
    is_cooperative: bool
    n_violations: int
    inference_time_ms: float


class RepairRequest(BaseModel):
    context: str = Field(..., min_length=1, max_length=2000)
    response: str = Field(..., min_length=1, max_length=1000)
    violation_type: str = Field(
        ..., pattern="^(quantity|quality|manner)$",
        description="The maxim to repair. Must be 'quantity', 'quality', or 'manner'. "
                    "Relation violations require FAISS retrieval and are not supported here."
    )


class RepairResponse(BaseModel):
    repaired_response: str
    is_fallback: bool                   # True = model was degenerate, returned original
    was_degenerate: bool                # True = model output was flagged as degenerate
    generation_method: str              # "beam" or "sample"
    inference_time_ms: float


class GenerateRequest(BaseModel):
    context: str = Field(
        ..., min_length=1, max_length=2000,
        description="Conversation history to respond to"
    )
    max_new_tokens: int = Field(
        default=100, ge=10, le=300,
        description="Maximum tokens to generate (10–300)"
    )


class GenerateResponse(BaseModel):
    generated_response: str
    inference_time_ms: float


class PipelineRequest(BaseModel):
    context: str = Field(..., min_length=1, max_length=2000)
    evidence: Optional[str] = Field(None, max_length=1000)
    max_new_tokens: int = Field(default=100, ge=10, le=300)

    model_config = {"json_schema_extra": {"example": {
        "context": "Have you seen any good films lately?",
        "evidence": None,
        "max_new_tokens": 100
    }}}


class PipelineStep(BaseModel):
    step: str
    details: dict


class PipelineResponse(BaseModel):
    final_response: str
    pipeline_steps: list[PipelineStep]  # Full audit trail
    is_cooperative: bool
    n_repairs_applied: int
    total_time_ms: float


# ── Model Registry ────────────────────────────────────────────────────────────
# Loaded at startup, kept in memory for all requests
_models: dict = {}


def _require_model(name: str):
    if name not in _models or _models[name] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{name}' is not loaded. Check /health for model status."
        )
    return _models[name]


# ── Lifespan: Model Loading on Startup ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at server startup; clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("GriceBench API Server — Loading models...")
    logger.info("=" * 60)

    # ── Load Detector ─────────────────────────────────────────────────
    try:
        from scripts.detector_inference import GriceBenchDetector
        _models["detector"] = GriceBenchDetector(
            model_path="best_model_v2.pt",
            temperatures_path="temperatures.json",
        )
        logger.info("✅ Detector loaded (DeBERTa-v3-base + Focal Loss)")
    except Exception as e:
        logger.error("❌ CRITICAL: Detector failed to load: %s", e)
        logger.error("   Detector is required for /detect, /repair, /pipeline endpoints.")
        raise

    # ── Load Repair Model ────────────────────────────────────────────
    try:
        from scripts.repair_inference_fixed import FixedRepairModel
        _models["repair"] = FixedRepairModel(
            model_path="models/repair/repair_model/",
        )
        logger.info("✅ Repair model loaded (T5-base, FixedRepairModel v2)")
    except Exception as e:
        logger.error("❌ CRITICAL: Repair model failed to load: %s", e)
        raise

    # ── Load DPO Generator (optional — pipeline still works without it) ──
    try:
        from scripts.generator_inference import GriceBenchGenerator
        _models["generator"] = GriceBenchGenerator(
            adapter_path="dpo_training_final_outcome/",
        )
        logger.info("✅ DPO Generator loaded (LoRA adapter)")
    except Exception as e:
        logger.warning("⚠️  Generator NOT loaded (non-critical): %s", e)
        logger.warning("   /generate and /pipeline endpoints will be unavailable.")
        _models["generator"] = None

    logger.info("🚀 Server ready. Listening on http://0.0.0.0:8000")

    yield  # ← Server is running here

    # Cleanup
    _models.clear()
    logger.info("Models unloaded. Server shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GriceBench API",
    description=(
        "Detect, repair, and generate Gricean-cooperative dialogue using the GriceBench pipeline.\n\n"
        "**Paper:** [EMNLP 2026] GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue\n\n"
        "**Models:** https://huggingface.co/PushkarPrabhath27"
    ),
    version="1.0.0",
    contact={"name": "Pushkar Prabhath"},
    license_info={"name": "Apache 2.0"},
    lifespan=lifespan,
)


# ── Middleware: Global Request Timer ─────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Log request timing for all endpoints."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(elapsed, 2))
    return response


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check with model status")
async def health_check():
    """
    Returns health status and which models are loaded.
    All three models loaded → system is fully operational.
    """
    return {
        "status": "healthy",
        "models": {
            "detector": "detector" in _models and _models["detector"] is not None,
            "repair":   "repair" in _models and _models["repair"] is not None,
            "generator": "generator" in _models and _models["generator"] is not None,
        },
        "version": "1.0.0",
        "endpoints": {
            "/detect":   "detector" in _models,
            "/repair":   "repair" in _models,
            "/generate": _models.get("generator") is not None,
            "/pipeline": _models.get("generator") is not None,
        }
    }


# ── /metrics ──────────────────────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus metrics endpoint. Scrape at http://localhost:8000/metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── POST /detect ──────────────────────────────────────────────────────────────
@app.post("/detect", response_model=DetectResponse, summary="Detect Gricean violations")
async def detect_violations(request: DetectRequest):
    """
    Detect Gricean maxim violations in a dialogue response.

    Returns binary violation flags and calibrated probabilities for all 4 maxims.
    Sub-100ms latency for typical inputs (DeBERTa on GPU).
    """
    detector = _require_model("detector")
    start = time.perf_counter()

    try:
        result = detector.detect(
            context=request.context,
            response=request.response,
            evidence=request.evidence or "",
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/detect", status="error").inc()
        logger.error("Detect error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000
    violations  = result.get("violations", {})
    probs       = result.get("probabilities", {})
    is_coop     = not any(violations.values())
    n_violations = sum(v for v in violations.values())

    # Update Prometheus metrics
    REQUEST_COUNT.labels(endpoint="/detect", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="/detect").observe(elapsed_ms / 1000)
    for maxim, violated in violations.items():
        if violated:
            VIOLATION_COUNTER.labels(maxim=maxim).inc()
    _update_cooperative_rate(is_coop)

    return DetectResponse(
        violations=violations,
        probabilities=probs,
        is_cooperative=is_coop,
        n_violations=n_violations,
        inference_time_ms=round(elapsed_ms, 2),
    )


# ── POST /repair ──────────────────────────────────────────────────────────────
@app.post("/repair", response_model=RepairResponse, summary="Repair a Gricean violation")
async def repair_response(request: RepairRequest):
    """
    Repair a single maxim violation using T5.

    Routing:
    - quantity/quality violations → beam search (repetition_penalty=1.5)
    - manner violations → temperature sampling (temperature=0.85, top_p=0.92)
    - relation violations → NOT SUPPORTED (requires FAISS retrieval)

    If the model output is degenerate (punctuation loops, trigram repetition),
    returns the original response with `is_fallback=True`.
    """
    repair_model = _require_model("repair")
    start = time.perf_counter()

    input_text = (
        f"fix {request.violation_type}: "
        f"[CONTEXT] {request.context} "
        f"[RESPONSE] {request.response}"
    )

    try:
        result = repair_model.repair(
            input_text=input_text,
            violation_type=request.violation_type,
            fallback_to_original=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/repair", status="error").inc()
        logger.error("Repair error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000

    if result["is_fallback"]:
        REPAIR_FALLBACK_COUNTER.inc()

    REQUEST_COUNT.labels(endpoint="/repair", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="/repair").observe(elapsed_ms / 1000)

    return RepairResponse(
        repaired_response=result["repaired_text"],
        is_fallback=result["is_fallback"],
        was_degenerate=result["is_degenerate"],
        generation_method=result["generation_method"],
        inference_time_ms=round(elapsed_ms, 2),
    )


# ── POST /generate ────────────────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse, summary="Generate cooperative response")
async def generate_response(request: GenerateRequest):
    """
    Generate a cooperative response using the DPO-trained generator.
    Returns raw generator output (not post-hoc repaired).
    Use /pipeline for the full detect+repair correction.
    """
    generator = _require_model("generator")
    start = time.perf_counter()

    try:
        result = generator.generate(
            context=request.context,
            max_new_tokens=request.max_new_tokens,
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/generate", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000
    REQUEST_COUNT.labels(endpoint="/generate", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="/generate").observe(elapsed_ms / 1000)

    return GenerateResponse(
        generated_response=result["text"],
        inference_time_ms=round(elapsed_ms, 2),
    )


# ── POST /pipeline ────────────────────────────────────────────────────────────
@app.post("/pipeline", response_model=PipelineResponse, summary="Full GriceBench pipeline")
async def full_pipeline(request: PipelineRequest):
    """
    Full GriceBench pipeline:
      1. Generate response (DPO generator)
      2. Detect violations (DeBERTa detector)
      3. Repair each detected violation (T5 repair model)
      4. Re-detect to confirm repair success
    
    Returns full audit trail of all pipeline steps.
    Requires all three models to be loaded.
    """
    detector  = _require_model("detector")
    repair    = _require_model("repair")
    generator = _require_model("generator")  # Will 503 if not loaded

    start = time.perf_counter()
    steps: list[PipelineStep] = []
    n_repairs = 0

    # ── Step 1: Generate ──────────────────────────────────────────────────────
    try:
        gen_result = generator.generate(request.context, request.max_new_tokens)
        current_response = gen_result["text"]
        steps.append(PipelineStep(
            step="generate",
            details={"response": current_response, "method": "dpo_generator"}
        ))
        logger.debug("Generate: %s", current_response[:80])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generator error: {e}")

    # ── Step 2: Detect violations ─────────────────────────────────────────────
    detect_result = detector.detect(
        context=request.context,
        response=current_response,
        evidence=request.evidence or "",
    )
    violations = detect_result.get("violations", {})
    steps.append(PipelineStep(
        step="detect",
        details={"violations": violations, "probabilities": detect_result.get("probabilities", {})}
    ))

    # ── Step 3: Repair each detected violation ────────────────────────────────
    REPAIRABLE_MAXIMS = ["quantity", "quality", "manner"]
    for maxim in REPAIRABLE_MAXIMS:
        if not violations.get(maxim, False):
            continue

        input_text = f"fix {maxim}: [CONTEXT] {request.context} [RESPONSE] {current_response}"
        try:
            repair_result = repair.repair(input_text, maxim, fallback_to_original=True)
            current_response = repair_result["repaired_text"]
            n_repairs += 1
            steps.append(PipelineStep(
                step=f"repair_{maxim}",
                details={
                    "repaired": current_response,
                    "is_fallback": repair_result["is_fallback"],
                    "was_degenerate": repair_result["is_degenerate"],
                    "method": repair_result["generation_method"],
                }
            ))
            if repair_result["is_fallback"]:
                REPAIR_FALLBACK_COUNTER.inc()
        except Exception as e:
            logger.warning("Repair failed for maxim='%s': %s", maxim, e)
            steps.append(PipelineStep(
                step=f"repair_{maxim}",
                details={"error": str(e), "skipped": True}
            ))

    # Relation: log as skipped (FAISS not integrated in this API)
    if violations.get("relation", False):
        steps.append(PipelineStep(
            step="repair_relation",
            details={"skipped": True, "reason": "Relation repair requires FAISS retrieval (not API-integrated)"}
        ))

    # ── Step 4: Re-detect ─────────────────────────────────────────────────────
    final_detect = detector.detect(
        context=request.context,
        response=current_response,
        evidence=request.evidence or "",
    )
    final_violations = final_detect.get("violations", {})
    is_cooperative = not any(final_violations.values())
    steps.append(PipelineStep(
        step="final_detect",
        details={"violations": final_violations, "is_cooperative": is_cooperative}
    ))

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Prometheus
    REQUEST_COUNT.labels(endpoint="/pipeline", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="/pipeline").observe(elapsed_ms / 1000)
    for maxim, violated in final_violations.items():
        if violated:
            VIOLATION_COUNTER.labels(maxim=maxim).inc()
    _update_cooperative_rate(is_cooperative)

    return PipelineResponse(
        final_response=current_response,
        pipeline_steps=steps,
        is_cooperative=is_cooperative,
        n_repairs_applied=n_repairs,
        total_time_ms=round(elapsed_ms, 2),
    )


# ── Startup / Shutdown log ────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_message():
    logger.info("API docs: http://localhost:8000/docs")


# ── Run directly for development ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (dev mode)
        log_level="info",
    )
