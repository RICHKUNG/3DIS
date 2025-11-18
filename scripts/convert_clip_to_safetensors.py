#!/usr/bin/env python
"""
Convert OpenAI CLIP model to safetensors format.

This solves the PyTorch 2.6 requirement issue by converting the model
to safetensors format, which doesn't have the security vulnerability.
"""

import argparse
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


def convert_model(
    source_model: str,
    output_dir: str,
    test_conversion: bool = True
):
    """
    Convert CLIP model to safetensors format.

    Args:
        source_model: Source model name or path
        output_dir: Output directory for converted model
        test_conversion: Test the converted model
    """
    print(f"Loading model from: {source_model}")
    model = CLIPModel.from_pretrained(source_model)
    processor = CLIPProcessor.from_pretrained(source_model)
    tokenizer = CLIPTokenizer.from_pretrained(source_model)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving model with safetensors to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Model converted successfully!")

    # Verify saved files
    saved_files = list(output_path.glob("*"))
    print(f"\nSaved files ({len(saved_files)}):")
    for f in sorted(saved_files):
        print(f"  - {f.name}")

    safetensor_files = list(output_path.glob("*.safetensors"))
    if safetensor_files:
        print(f"\n✅ Found {len(safetensor_files)} safetensors files")
    else:
        print("\n⚠️ WARNING: No safetensors files found!")

    # Test conversion
    if test_conversion:
        print("\nTesting converted model...")
        test_model = CLIPModel.from_pretrained(output_dir)
        test_processor = CLIPProcessor.from_pretrained(output_dir)
        test_tokenizer = CLIPTokenizer.from_pretrained(output_dir)

        # Quick test
        import torch
        text = test_tokenizer(["a photo of a cat"], padding=True, return_tensors="pt")
        with torch.no_grad():
            text_features = test_model.get_text_features(**text)

        print(f"✅ Model test passed! Feature dim: {text_features.shape[-1]}")

    print(f"\n✅ All done! Use this path in your config:")
    print(f"   siglip_model: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Convert CLIP model to safetensors")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="Source model name or path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/clip-vit-large-safetensors",
        help="Output directory"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip testing the converted model"
    )

    args = parser.parse_args()

    convert_model(
        source_model=args.model,
        output_dir=args.output,
        test_conversion=not args.no_test
    )


if __name__ == "__main__":
    main()
