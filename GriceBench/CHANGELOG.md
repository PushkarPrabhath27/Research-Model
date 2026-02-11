# Changelog

All notable changes to GriceBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-23

### Added
- **Part 1: Relation Repair System**
  - Retrieval-augmented repair for Relation violations
  - FAISS-based evidence retrieval
  - Response corpus from 50K examples
  
- **Part 2: Human Evaluation Framework**
  - Gradio web interface for annotation
  - CLI interface with 5-dimensional rubric
  - Krippendorff's α inter-annotator agreement
  - Blinded sample preparation

- **Part 3: Baseline Comparisons**
  - Mistral-7B baseline (89.1% cooperative)
  - Qwen2.5-7B baseline (84.2% cooperative)
  - Automated evaluation pipeline

- **Part 4: Ablation Studies**
  - Component ablation (detector, repair, DPO)
  - Threshold sensitivity analysis (0.3-0.7)
  - Maxim importance analysis
  - Results: Full system 95% vs baseline 83.8%

- **Part 5: Error Analysis**
  - Confusion matrices for all 4 maxims
  - Hardest examples identification
  - Failure mode categorization
  - 94.2% exact match accuracy

- **Part 6: Documentation & Reproducibility**
  - Production-quality README (644 lines)
  - Model cards for detector, repair, DPO
  - Complete dataset documentation
  - Pinned dependency versions
  - Experiment runner scripts

- **Part 7: Production Infrastructure**
  - GitHub Actions CI/CD pipeline
  - Docker & docker-compose deployment
  - Pre-commit hooks (black, isort, flake8)
  - Unit and integration tests
  - Contributing guidelines

- **Part 8: Performance Optimization**
  - Memory and latency profiling tools
  - INT8/FP16 quantization support
  - ONNX export for deployment
  - Comprehensive benchmark suite
  - Performance optimization guide

- **Part 9: API & Production**
  - Production FastAPI server
  - Prometheus metrics
  - Grafana dashboard
  - Authentication & rate limiting
  - Complete API documentation
  - Troubleshooting guide
  - Production deployment checklist

### Performance
- Detector F1: 0.968 (macro-average)
- Repair BLEU: 46.8 (overall)
- Full system cooperative rate: 95.0%
- API latency: <100ms p95

### Infrastructure
- 48+ production-ready files
- 20+ Python scripts
- 15 documentation files
- Complete testing framework
- End-to-end CI/CD

---

## [Unreleased]

### Planned
- Multi-GPU inference support
- TensorRT optimization
- Kubernetes deployment examples
- Extended language support
- Real-time streaming API

---

## Release Notes Template

```
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security patches
```

---

**Versioning Scheme:**
- **MAJOR** version: Incompatible API changes
- **MINOR** version: Backwards-compatible functionality additions
- **PATCH** version: Backwards-compatible bug fixes
