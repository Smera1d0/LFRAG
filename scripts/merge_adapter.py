import argparse
import torch
from peft import PeftModel
from transformers import Qwen2_5_VLModel, AutoProcessor
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into the base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the merged model")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model}...")
    base = Qwen2_5_VLModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging and unloading...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    merged.save_pretrained(args.output_dir)

    adapter_path = Path(args.adapter)
    output_path = Path(args.output_dir)
    tokenizer_files = [
        "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
        "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
        "chat_template.json", "video_preprocessor_config.json", "generation_config.json",
    ]
    for fname in tokenizer_files:
        src = adapter_path / fname
        if not src.exists():
            src = Path(args.base_model) / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)

    print("Done.")


if __name__ == "__main__":
    main()
