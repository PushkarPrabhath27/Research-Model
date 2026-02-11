"""
Production-ready FastAPI server for GriceBench
Provides REST endpoints for violation detection, repair, and generation
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import torch
from transformers import AutoTokenizer
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('gricebench_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('gricebench_request_duration_seconds', 'Request latency', ['endpoint'])
DETECTION_COUNT = Counter('gricebench_detections_total', 'Total detections', ['maxim'])

# Initialize FastAPI app
app = FastAPI(
    title="GriceBench API",
    description="Production API for Gricean Maxim violation detection, repair, and generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class DetectionRequest(BaseModel):
    context: str = Field(..., description="Dialogue context", min_length=1)
    response: str = Field(..., description="Response to evaluate", min_length=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Detection threshold")

class DetectionResponse(BaseModel):
    violations: Dict[str, bool] = Field(..., description="Detected violations")
    probabilities: Dict[str, float] = Field(..., description="Violation probabilities")
    cooperative: bool = Field(..., description="Whether response is cooperative")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")

class RepairRequest(BaseModel):
    context: str = Field(..., description="Dialogue context")
    response: str = Field(..., description="Response to repair")
    violations: List[str] = Field(..., description="Detected violations to fix")

class RepairResponse(BaseModel):
    repaired_response: str = Field(..., description="Repaired response")
    latency_ms: float = Field(..., description="Repair latency in milliseconds")

class GenerationRequest(BaseModel):
    context: str = Field(..., description="Dialogue context")
    max_length: int = Field(100, ge=10, le=500, description="Max response length")

class GenerationResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    gpu_available: bool

# Global model storage
class ModelManager:
    def __init__(self):
        self.detector = None
        self.detector_tokenizer = None
        self.repair = None
        self.repair_tokenizer = None
        self.generator = None
        self.generator_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_models(self):
        """Load all models"""
        logger.info("Loading models...")
        
        # Load detector
        try:
            from scripts.train_detector import ViolationDetector
            self.detector = ViolationDetector.from_pretrained("models/detector")
            self.detector_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            self.detector = self.detector.to(self.device)
            self.detector.eval()
            logger.info("✅ Detector loaded")
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
        
        # Load repair (optional)
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.repair = T5ForConditionalGeneration.from_pretrained("models/repair")
            self.repair_tokenizer = T5Tokenizer.from_pretrained("models/repair")
            self.repair = self.repair.to(self.device)
            self.repair.eval()
            logger.info("✅ Repair loaded")
        except Exception as e:
            logger.warning(f"Repair model not loaded: {e}")
        
        # Load generator (optional)
        try:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
            self.generator = PeftModel.from_pretrained(base_model, "models/dpo")
            self.generator_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
            self.generator = self.generator.to(self.device)
            self.generator.eval()
            logger.info("✅ Generator loaded")
        except Exception as e:
            logger.warning(f"Generator model not loaded: {e}")
        
        logger.info(f"All models loaded on {self.device}")

# Initialize model manager
model_manager = ModelManager()

# API key authentication (simple example)
async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key (implement proper authentication in production)"""
    # For production, use proper auth like JWT tokens
    if x_api_key != "your-api-key-here":  # Replace with env variable
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting GriceBench API server...")
    model_manager.load_models()
    logger.info("Server ready!")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "detector": model_manager.detector is not None,
            "repair": model_manager.repair is not None,
            "generator": model_manager.generator is not None
        },
        gpu_available=torch.cuda.is_available()
    )

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

# Detection endpoint
@app.post("/detect", response_model=DetectionResponse)
async def detect_violations(request: DetectionRequest):
    """Detect Gricean maxim violations"""
    start_time = time.time()
    
    try:
        if model_manager.detector is None:
            raise HTTPException(status_code=503, detail="Detector model not loaded")
        
        # Prepare input
        input_text = f"[CONTEXT] {request.context} [RESPONSE] {request.response}"
        inputs = model_manager.detector_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model_manager.device)
        
        # Inference
        with torch.no_grad():
            outputs = model_manager.detector(inputs['input_ids'], inputs['attention_mask'])
        
        probs = outputs['probs'].cpu().numpy()[0]
        violations = {
            "quantity": bool(probs[0] > request.threshold),
            "quality": bool(probs[1] > request.threshold),
            "relation": bool(probs[2] > request.threshold),
            "manner": bool(probs[3] > request.threshold)
        }
        
        probabilities = {
            "quantity": float(probs[0]),
            "quality": float(probs[1]),
            "relation": float(probs[2]),
            "manner": float(probs[3])
        }
        
        cooperative = not any(violations.values())
        
        # Metrics
        for maxim, violated in violations.items():
            if violated:
                DETECTION_COUNT.labels(maxim=maxim).inc()
        
        latency_ms = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint='detect').observe(time.time() - start_time)
        REQUEST_COUNT.labels(endpoint='detect', status='success').inc()
        
        return DetectionResponse(
            violations=violations,
            probabilities=probabilities,
            cooperative=cooperative,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='detect', status='error').inc()
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Repair endpoint
@app.post("/repair", response_model=RepairResponse)
async def repair_response(request: RepairRequest):
    """Repair a response with violations"""
    start_time = time.time()
    
    try:
        if model_manager.repair is None:
            raise HTTPException(status_code=503, detail="Repair model not loaded")
        
        # Prepare input
        violation_str = ", ".join(request.violations).upper()
        input_text = f"repair violation: {violation_str} context: {request.context} response: {request.response}"
        
        inputs = model_manager.repair_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model_manager.device)
        
        # Generate repair
        outputs = model_manager.repair.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True
        )
        
        repaired = model_manager.repair_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        latency_ms = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint='repair').observe(time.time() - start_time)
        REQUEST_COUNT.labels(endpoint='repair', status='success').inc()
        
        return RepairResponse(
            repaired_response=repaired,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='repair', status='error').inc()
        logger.error(f"Repair error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Generation endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate_response(request: GenerationRequest):
    """Generate a cooperative response"""
    start_time = time.time()
    
    try:
        if model_manager.generator is None:
            raise HTTPException(status_code=503, detail="Generator model not loaded")
        
        # Prepare input
        prompt = f"Context: {request.context}\\nGenerate cooperative response:"
        inputs = model_manager.generator_tokenizer(prompt, return_tensors="pt").to(model_manager.device)
        
        # Generate
        outputs = model_manager.generator.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=model_manager.generator_tokenizer.eos_token_id
        )
        
        response_text = model_manager.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract response (remove prompt)
        response_text = response_text.split("cooperative response:")[-1].strip()
        
        latency_ms = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint='generate').observe(time.time() - start_time)
        REQUEST_COUNT.labels(endpoint='generate', status='success').inc()
        
        return GenerationResponse(
            response=response_text,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='generate', status='error').inc()
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        log_level="info"
    )
