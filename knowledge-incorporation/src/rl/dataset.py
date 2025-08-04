# knowledge-incorporation/src/rl/dataset.py
import json
import random
from datasets import load_dataset

def build_dataset(dataset_name: str = "squad", split: str = "train", use_synthetic: bool = True):
    """
    Loads the SQuAD dataset or synthetic data and prepares it for the RL training loop.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (e.g., 'train', 'validation').
        use_synthetic (bool): Whether to use synthetic data instead of raw SQuAD.

    Returns:
        A list of data items or Hugging Face Dataset object.
    """
    if use_synthetic:
        # Load the synthetic data generated in Phase 1-3
        import os
        synthetic_path = "knowledge-incorporation/mlx_experiments/data/synthetic_data/train/squad_train_mlx_generated.json"
        full_path = os.path.join(os.getcwd(), synthetic_path)
        
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                synthetic_data = json.load(f)
            print(f"Loaded {len(synthetic_data)} synthetic articles from {synthetic_path}")
            return synthetic_data
        else:
            print(f"Synthetic data not found at {full_path}, falling back to SQuAD...")
            return load_dataset(dataset_name, split=split)
    else:
        # Load the original SQuAD dataset
        return load_dataset(dataset_name, split=split)

def get_squad_prompts(dataset, num_samples: int = 100):
    """
    Extracts prompts from the SQuAD dataset or synthetic data.

    Args:
        dataset: A Hugging Face Dataset object (SQuAD) or list of synthetic data.
        num_samples (int): The number of samples to extract.

    Returns:
        A list of tuples: (prompt_string, gold_answer).
    """
    prompt_answer_pairs = []
    
    # Handle synthetic data format
    if isinstance(dataset, list):
        # Flatten all questions from all articles
        all_questions = []
        for article in dataset:
            for qa in article.get('questions', []):
                all_questions.append({
                    'title': article['title'],
                    'context': article['context'],
                    'question': qa['question'],
                    'answer': qa['answer']
                })
        
        # Randomly sample from all available questions
        sampled_questions = random.sample(all_questions, min(num_samples, len(all_questions)))
        
        for item in sampled_questions:
            prompt = f"Title: {item['title']}\n\nContext: {item['context']}\n\n---\n\nQuestion: {item['question']}\n\nAnswer:"
            gold_answer = item['answer']
            prompt_answer_pairs.append((prompt, gold_answer))
            
        print(f"Sampled {len(prompt_answer_pairs)} prompt-answer pairs from {len(all_questions)} total questions across {len(dataset)} articles")
        
    else:
        # Handle original SQuAD format
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            prompt = f"Title: {item['title']}\n\nContext: {item['context']}\n\n---\n\nQuestion: {item['question']}\n\nAnswer:"
            gold_answer = item['answers']['text'][0] if item['answers']['text'] else ""
            prompt_answer_pairs.append((prompt, gold_answer))
    
    return prompt_answer_pairs

if __name__ == '__main__':
    # Example of how to use the functions
    squad_dataset = build_dataset()
    prompts = get_squad_prompts(squad_dataset, num_samples=5)
    
    print("Successfully loaded the SQuAD dataset.")
    print(f"Extracted {len(prompts)} sample prompts.")
    print("\nExample Prompt:")
    print(prompts[0])
