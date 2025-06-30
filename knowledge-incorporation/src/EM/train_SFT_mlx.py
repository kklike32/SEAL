
import argparse
from mlx_lm.tuner import train
from mlx_lm.tuner.trainer import TrainingArgs
from mlx_lm import load
import datasets

def main():
    """
    Main function to run the Supervised Fine-Tuning (SFT) process.
    """
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with MLX.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo ID of the base model to fine-tune.")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to the training dataset in .jsonl format.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of layers to apply LoRA to.")

    args = parser.parse_args()

    print(f"Loading base model: {args.model}")
    model, tokenizer = load(args.model)

    # The build_SFT_dataset.py script creates a 'text' field.
    # We need to tell the trainer to use this field.
    training_args = TrainingArgs(
        batch_size=1,
        iters=100, # A reasonable number of iterations
        val_batches=1,
        steps_per_report=10,
        steps_per_eval=20,
        steps_per_save=20,
        adapter_file="adapters.npz",
        learning_rate=args.lr,
        lora_layers=args.lora_layers,
        train_dataset_key="text" # Specify the key for the training text
    )

    print(f"Loading training data from: {args.train_dataset}")
    train_dataset = datasets.load_dataset("json", data_files=args.train_dataset)

    print("Starting Supervised Fine-Tuning...")
    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset["train"],
    )

    print(f"Fine-tuning complete. Adapter saved in {args.output_dir}")

if __name__ == "__main__":
    main()
