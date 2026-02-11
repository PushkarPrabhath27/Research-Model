"""
Export GriceBench models to ONNX format
Enables deployment on ONNX Runtime, TensorRT, OpenVINO
"""

import argparse
import torch
import onnx
from pathlib import Path
from transformers import AutoTokenizer


def export_to_onnx(model, tokenizer, output_path, opset_version=14):
    """Export PyTorch model to ONNX"""
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    # Create dummy inputs
    batch_size = 1
    seq_length = 128
    
    dummy_input = tokenizer(
        "Sample text for ONNX export",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_length
    )
    
    dummy_input_ids = dummy_input['input_ids']
    dummy_attention_mask = dummy_input['attention_mask']
    
    # Export
    model.eval()
    
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits', 'probs'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'},
            'probs': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported to {output_path}")


def verify_onnx_model(onnx_path):
    """Verify ONNX model is valid"""
    print(f"\\nVerifying ONNX model...")
    
    # Load and check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ ONNX model is valid")
    
    # Print model info
    print(f"\\nModel info:")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")


def test_onnx_inference(onnx_path, tokenizer):
    """Test ONNX model inference"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️ onnxruntime not installed, skipping inference test")
        return
    
    print(f"\\nTesting ONNX inference...")
    
    # Create session
    session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']  # Use CPU for compatibility
    )
    
    # Prepare input
    text = "Test sentence for ONNX inference"
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    
    ort_inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    
    # Run inference
    outputs = session.run(None, ort_inputs)
    
    print(f"✅ ONNX inference successful!")
    print(f"   Output shapes: {[out.shape for out in outputs]}")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Model to export")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    parser.add_argument("--test", action="store_true", help="Test inference")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ONNX EXPORT")
    print("="*60)
    
    # Load model
    print(f"Loading model from {args.model}...")
    
    if "detector" in args.model.lower():
        from scripts.train_detector import ViolationDetector
        model = ViolationDetector.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_to_onnx(model, tokenizer, str(output_path), args.opset)
    
    # Verify
    if args.verify:
        verify_onnx_model(str(output_path))
    
    # Test
    if args.test:
        test_onnx_inference(str(output_path), tokenizer)
    
    print(f"\\n✅ Export complete!")


if __name__ == "__main__":
    main()
