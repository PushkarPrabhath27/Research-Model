# GriceBench API Documentation

## Overview

The GriceBench API provides REST endpoints for violation detection, repair, and generation.

**Base URL:** `http://localhost:8000`

---

## Authentication

All endpoints require an API key passed in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key-here" http://localhost:8000/detect
```

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "detector": true,
    "repair": true,
    "generator": true
  },
  "gpu_available": true
}
```

---

### 2. Detect Violations

**POST** `/detect`

Detect Gricean maxim violations in a response.

**Request Body:**
```json
{
  "context": "What is quantum computing?",
  "response": "Quantum computing uses qubits for processing.",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "violations": {
    "quantity": false,
    "quality": false,
    "relation": false,
    "manner": false
  },
  "probabilities": {
    "quantity": 0.023,
    "quality": 0.015,
    "relation": 0.008,
    "manner": 0.031
  },
  "cooperative": true,
  "latency_ms": 12.5
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/detect \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-api-key-here" \\
  -d '{
    "context": "What is AI?",
    "response": "AI is artificial intelligence",
    "threshold": 0.5
  }'
```

---

### 3. Repair Response

**POST** `/repair`

Repair a response with violations.

**Request Body:**
```json
{
  "context": "What is quantum computing?",
  "response": "Quantum comp. uses q-bits.",
  "violations": ["manner"]
}
```

**Response:**
```json
{
  "repaired_response": "Quantum computing uses quantum bits (qubits).",
  "latency_ms": 45.2
}
```

---

### 4. Generate Response

**POST** `/generate`

Generate a cooperative response.

**Request Body:**
```json
{
  "context": "What is your favorite movie?",
  "max_length": 100
}
```

**Response:**
```json
{
  "response": "I enjoy science fiction films like Inception and Interstellar. They explore fascinating concepts about reality and time.",
  "latency_ms": 120.8
}
```

---

### 5. Metrics

**GET** `/metrics`

Prometheus metrics endpoint.

**Response:** Plain text Prometheus format

---

## Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid request parameters"
}
```

**403 Forbidden:**
```json
{
  "detail": "Invalid API key"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Detector model not loaded"
}
```

---

## Rate Limiting

- **Default:** 100 requests/minute per API key
- **Burst:** 200 requests/minute

---

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

**Last Updated:** 2026-01-23
