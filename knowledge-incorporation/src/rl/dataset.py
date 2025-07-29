import json
from datasets import load_dataset

def build_dataset(dataset_name: str = "squad", split: str = "train"):
    """
    Loads the SQuAD dataset and prepares it for the RL training loop.

    For now, this function will just load the dataset from Hugging Face.
    We will extend it to load our locally generated data.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (e.g., 'train', 'validation').

    Returns:
        A Hugging Face Dataset object.
    """
    # For now, we load the original SQuAD dataset.
    # We will modify this to use our generated data with completions.
    return load_dataset(dataset_name, split=split)

def get_squad_prompts(dataset, num_samples: int = 100):
    """
    Extracts prompts from the SQuAD dataset.

    Args:
        dataset: A Hugging Face Dataset object (SQuAD).
        num_samples (int): The number of samples to extract.

    Returns:
        A list of prompts (strings).
    """
    prompts = []
    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        # We will use the same prompt format as in the previous phases
        prompt = f"Title: {item['title']}\n\nContext: {item['context']}\n\n---\n\nQuestion: {item['question']}\n\nAnswer:"
        prompts.append(prompt)
    return prompts

if __name__ == '__main__':
    # Example of how to use the functions
    squad_dataset = build_dataset()
    prompts = get_squad_prompts(squad_dataset, num_samples=5)
    
    print("Successfully loaded the SQuAD dataset.")
    print(f"Extracted {len(prompts)} sample prompts.")
    print("\nExample Prompt:")
    print(prompts[0])
