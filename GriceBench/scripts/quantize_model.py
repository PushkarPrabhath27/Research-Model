"""
Quantize GriceBench models to INT8 for faster inference
Applies dynamic or static quantization
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer


def dynamic_quantization(model, output_path):
    """Apply dynamic INT8 quantization"""
    print("Applying dynamic quantization...")
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Save
    torch.save(quantized_model.state_dict(), output_path)
    print(f"✅ Quantized model saved to {output_path}")
    
    return quantized_model


def static_quantization(model, tokenizer, calibration_data, output_path):
    """Apply static INT8 quantization with calibration"""
    print("Applying static quantization...")
    
    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate
    print("Calibrating on calibration data...")
    model.eval()
    
    with torch.no_grad():
        for i, text in enumerate(calibration_data):
            if i % 100 == 0:
                print(f"  Calibrated {i}/{len(calibration_data)} samples")
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            _ = model(**inputs)
    
    # Convert
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # Save
    torch.save(quantized_model.state_dict(), output_path)
    print(f"✅ Quantized model saved to {output_path}")
    
    return quantized_model


def compare_models(original_model, quantized_model, tokenizer, test_samples):
    """Compare original vs quantized model"""
    print("\\nComparing models...")
    
    original_model.eval()
    quantized_model.eval()
    
    # Test on sample inputs
    test_text = test_samples[0]
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)
    
    # Original
    with torch.no_grad():
        original_output = original_model(**inputs)
    
    # Quantized
    with torch.no_grad():
        quantized_output = quantized_model(**inputs)
    
    # Compare outputs (assuming classification task)
    if hasattr(original_output, 'logits'):
        original_logits = original_output.logits
        quantized_logits = quantized_output.logits
    else:
        original_logits = original_output[0]
        quantized_logits = quantized_output[0]
    
    # Calculate difference
    diff = torch.abs(original_logits - quantized_logits).mean().item()
    
    print(f"\\nOutput difference: {diff:.6f}")
    print(f"Original shape: {original_logits.shape}")
    print(f"Quantized shape: {quantized_logits.shape}")


def main():
    parser = argparse.ArgumentParser(description="Quantize GriceBench models")
    parser.add_argument("--model", type=str, required=True, help="Model to quantize")
    parser.add_argument("--method", type=str, default="dynamic", choices=["dynamic", "static"])
    parser.add_argument("--calibration_samples", type=int, default=500, help="Num calibration samples for static quantization")
    parser.add_argument("--output", type=str, required=True, help="Output path for quantized model")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL QUANTIZATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    
    # Load model
    print(f"\\nLoading model...")
    if args.model == "detector":
        from scripts.train_detector import ViolationDetector
        model = ViolationDetector.from_pretrained("models/detector")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    model.eval()
    
    # Quantize
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.method == "dynamic":
        quantized_model = dynamic_quantization(model, output_path)
    else:
        # Load calibration data
        calibration_data = ["Sample text " + str(i) for i in range(args.calibration_samples)]
        quantized_model = static_quantization(model, tokenizer, calibration_data, output_path)
    
    # Compare
    test_samples = ["This is a test sentence"]
    compare_models(model, quantized_model, tokenizer, test_samples)
    
    print(f"\\n✅ Quantization complete!")
    print(f"   Model saved to: {output_path}")


if __name__ == "__main__":
    main()
