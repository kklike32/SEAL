import argparse
import os
from mlx_lm import lora, load

def parse_args():
    """
    Parses command-line arguments, mirroring the original train_SFT.py script
    for consistency and ease of use.
    """
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with MLX LoRA")
    
    # --- Paths and Model ---
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo ID of the base model to fine-tune (e.g., 'mlx-community/Meta-Llama-3-8B-Instruct-MLX').")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset in .jsonl format (output of build_SFT_dataset.py).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final, merged model.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the AdamW optimizer.")
    
    # --- LoRA Specific Hyperparameters ---
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA matrices.")
    parser.add_argument("--lora_alpha", type=int, default=128, help="The alpha for the LoRA scaling (often 2x rank).")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")
    parser.add_argument("--lora_layers", type=int, default=16, help="Number of layers to apply LoRA to, from the top down.")
    
    # --- Other Training Args ---
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--iters_per_report", type=int, default=10, help="Number of iterations between logging training loss.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 20)
    print("Starting MLX Supervised Fine-Tuning (SFT)")
    print(f"         Model: {args.model}")
    print(f"Training data: {args.train_file}")
    print(f"   Output dir: {args.output_dir}")
    print("=" * 20)

    # The fine_tune function handles loading the model, preparing the data,
    # and running the training loop in a single, convenient call.
    lora.fine_tune(
        model=args.model,
        # The SFT build script produces a file with a 'text' column, which is the default
        # expected by the fine_tune function.
        train_dataset_path=args.train_file,
        # All hyperparameters are passed directly.
        lora_layers=args.lora_layers,
        lora_parameters={
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "scale": 10.0,
        },
        batch_size=args.batch_size,
        iters=0, # iters=0 with epochs > 0 means train for that many epochs
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        # --- Crucially, save the merged model at the end ---
        save_path=args.output_dir,
        merge=True, # This fuses the adapter and saves the full model
    )

    print("\n" + "=" * 20)
    print("SFT complete.")
    print(f"Final merged model saved to: {args.output_dir}")
    print("=" * 20)

if __name__ == "__main__":
    main()