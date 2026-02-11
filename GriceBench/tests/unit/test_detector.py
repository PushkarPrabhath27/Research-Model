"""
Unit tests for violation detector
"""

import pytest
import torch
from scripts.train_detector import ViolationDetector


@pytest.fixture
def detector():
    """Create detector instance for testing"""
    return ViolationDetector("microsoft/deberta-v3-base", num_labels=4)


def test_detector_initialization(detector):
    """Test detector model initializes correctly"""
    assert detector is not None
    assert detector.classifier.out_features == 4


def test_detector_forward_pass(detector):
    """Test detector produces valid outputs"""
    batch_size = 2
    seq_length = 128
    
    # Create dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    outputs = detector(input_ids, attention_mask)
    
    assert 'logits' in outputs
    assert 'probs' in outputs
    assert outputs['logits'].shape == (batch_size, 4)
    assert outputs['probs'].shape == (batch_size, 4)
    assert torch.all(outputs['probs'] >= 0) and torch.all(outputs['probs'] <= 1)


def test_detector_output_range(detector):
    """Test probabilities are in valid range"""
    input_ids = torch.randint(0, 1000, (1, 64))
    attention_mask = torch.ones(1, 64)
    
    with torch.no_grad():
        outputs = detector(input_ids, attention_mask)
    
    probs = outputs['probs']
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
