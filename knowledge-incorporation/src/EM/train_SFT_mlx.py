# knowledge-incorporation/src/EM/train_SFT_mlx.py

import argparse
import os
from pathlib import Path

import mlx.optimizers as optim
from datasets import load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load, save_config
from mlx_lm.tuner.trainer import iterate_batches as default_iter

def encode_dataset(hf_ds, tok, max_len):
    encoded = []
    for ex in hf_ds:
        p_ids = tok.encode(ex["prompt"])
        c_ids = tok.encode(ex["completion"])
        ids   = (p_ids + c_ids)[:max_len]
        offset = len(p_ids)              # <- mask prompt!
        encoded.append((ids, offset))
    return encoded

def parse_args():
    """
    Parses command-line arguments, mirroring the original train_SFT.py script
    for consistency and ease of use.
    """
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with MLX LoRA")
    
    # --- Paths and Model ---
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo ID of the base model to fine-tune.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset in .jsonl format.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned LoRA adapter files.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the AdamW optimizer.")
    
    # --- LoRA Specific Hyperparameters ---
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA matrices.")
    parser.add_argument("--lora_alpha", type=int, default=128, help="The alpha for the LoRA scaling (often 2x rank).")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")
    parser.add_argument("--lora_layers", type=int, default=16, help="Number of layers to apply LoRA to.")
    
    # --- Other Training Args ---
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--steps_per_report", type=int, default=10, help="Number of iterations between logging training loss.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Provide clear feedback to the user about the upcoming training run
    print("\n" + "=" * 50)
    print("      Starting MLX Supervised Fine-Tuning (SFT)")
    print("=" * 50)
    print(f"| Base Model:         {args.model}")
    print(f"| Training Data:      {args.train_file}")
    print(f"| Adapter Output Dir: {args.output_dir}")
    print(f"|--------------------------------------------------")
    print(f"| Epochs:             {args.num_train_epochs}")
    print(f"| Learning Rate:      {args.learning_rate}")
    print(f"| Batch Size:         {args.batch_size}")
    print(f"|--------------------------------------------------")
    print(f"| LoRA Layers:        {args.lora_layers}")
    print(f"| LoRA Rank:          {args.lora_rank}")
    print(f"| LoRA Alpha:         {args.lora_alpha}")
    print("=" * 50 + "\n")

    # 1. Load Model and Tokenizer
    model, tokenizer = load(args.model)

    # 2. Load and Prepare Dataset
    train_dataset = load_dataset("json", data_files={"train": args.train_file}, split="train")

    # 3. Apply LoRA layers to the model
    model.freeze()
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "scale": 10.0
    }
    linear_to_lora_layers(model, args.lora_layers, lora_config)

    # 4. Define Training Arguments
    # Calculate total iterations based on epochs and dataset size
    total_iters = (len(train_dataset) // args.batch_size) * args.num_train_epochs
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=total_iters,
        val_batches=0,
        steps_per_report=args.steps_per_report,
        steps_per_eval=total_iters + 1, # De-facto disable evaluation during training
        steps_per_save=total_iters, # Save only once at the end
        adapter_file=str(adapter_file),
        max_seq_length=args.max_seq_length,
    )

    # 5. Start Training by calling the correct low-level train function
    print("Starting SFT training...")
    encoded_train = encode_dataset(train_dataset, tokenizer, args.max_seq_length)
    train(
        model=model,
        args=training_args,
        optimizer=optim.AdamW(learning_rate=args.learning_rate),
        train_dataset=encoded_train,
        iterate_batches=default_iter,
        val_dataset=encoded_train[:1],
    )

    # 6. Save the configuration and tokenizer alongside the adapter
    # This makes the adapter portable and easy to use later.
    save_config(vars(args), output_dir / "adapter_config.json")
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 50)
    print("SFT complete.")
    print(f"Final adapter saved to: {args.output_dir}")
    print("You can now fuse this adapter with the base model to create a final merged model.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
