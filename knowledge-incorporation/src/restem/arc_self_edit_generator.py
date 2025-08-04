# knowledge-incorporation/src/restem/arc_self_edit_generator.py

"""
ARC Self-Edit Generation for RestEM

This module generates multiple solution attempts for ARC tasks,
following the original SEAL few-shot methodology exactly.
"""

import os
import json
import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from src.utils_mlx import generate_mlx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Import shared model manager
from .shared_model import get_shared_model


@dataclass
class ARCExample:
    """Represents an ARC input-output example."""
    input_grid: np.ndarray
    output_grid: Optional[np.ndarray] = None


@dataclass 
class ARCTask:
    """Represents an ARC task with training and test examples."""
    name: str
    train_examples: List[ARCExample]
    test_examples: List[ARCExample]
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize task to dictionary format."""
        return {
            'train': [
                {
                    'input': example.input_grid.tolist(),
                    'output': example.output_grid.tolist() if example.output_grid is not None else None
                }
                for example in self.train_examples
            ],
            'test': [
                {
                    'input': example.input_grid.tolist(),
                    'output': example.output_grid.tolist() if example.output_grid is not None else None
                }
                for example in self.test_examples
            ]
        }


class ARCSelfEditGenerator:
    """
    Generates training configurations for ARC tasks following SEAL methodology.
    
    CORRECTED: This implements the configuration-based approach from the original paper:
    1. Generate JSON configurations specifying augmentation strategies
    2. Each config contains data generation and training parameters
    3. NO solution text generation - only configuration generation
    """
    
    def __init__(self, model_name: str, use_mlx: bool = True):
        self.model_name = model_name
        self.use_mlx = use_mlx
        
        if use_mlx and MLX_AVAILABLE:
            # Use shared model instead of loading our own
            self.model, self.tokenizer = get_shared_model(model_name)
            logging.info(f"Initialized ARCSelfEditGenerator with shared {model_name} (MLX: {use_mlx})")
        elif HF_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.use_mlx = False
            logging.info(f"Initialized ARCSelfEditGenerator with {model_name} (MLX: False)")
        else:
            raise ImportError("Neither MLX nor HuggingFace transformers are available")
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Original SEAL prompts
        self.system_message = "You are a helpful assistant that provide the correct output for the given task immediately."
        
        self.self_edit_prompt = """
You are configuring a model training pipeline by selecting from predefined tools.

You must make two decisions:

1. **Data Generation Tools** — For each of the following, choose true or false:
    - use_basic_augmentations
    - use_size_augmentations
    - use_chain_augmentations
    - use_repeat_augmentations

2. **Training Configuration** — Choose one of:
    - "train_using_all_tokens"
    - "train_using_output_tokens"

Also specify:
    - learning_rate (float)
    - num_train_epochs (integer)

### Output Format

Respond with a valid JSON object. Do not include any explanation, markdown, or extra text. Use lowercase `true`/`false` for booleans and ensure correct JSON syntax.

Example output:

{
  "data_generation": {
    "use_basic_augmentations": ...,
    "use_size_augmentations": ...,
    "use_chain_augmentations": ...,
    "use_repeat_augmentations": ...
  },
  "training": {
    "strategy": ...,
    "learning_rate": ...,
    "num_train_epochs": ...
  }
}
"""
            
        logging.info(f"Initialized ARCSelfEditGenerator with {model_name} (MLX: {self.use_mlx})")
    
    def generate_self_edit_configs(
        self, 
        task: ARCTask, 
        n_configs: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple training configurations for a task.
        
        CORRECTED: This generates JSON configurations specifying augmentation strategies
        and training parameters, not solution text. This is the actual SEAL methodology.
        """
        configs = []
        explored_configs = set()
        
        while len(configs) < n_configs:
            config_data = self.generate_config_for_task(task)
            
            if config_data is None:
                continue
                
            # Check for duplicates
            config_key = self._config_to_key(config_data["config"])
            if config_key in explored_configs:
                continue
                
            explored_configs.add(config_key)
            configs.append(config_data)
            
            logging.debug(f"Generated config {len(configs)}: {config_data['config']}")
        
        logging.info(f"Generated {len(configs)} configurations for task {task.name}")
        return configs
    def generate_config_for_task(self, task: ARCTask) -> Dict[str, Any]:
        """
        Generate a single training configuration for an ARC task.
        
        CORRECTED: This generates JSON configs specifying augmentation strategies,
        not solution text. This is the core SEAL methodology.
        """
        formatted_examples = self._format_task_examples(task)
        user_message = formatted_examples + "------\n\n" + self.self_edit_prompt
        
        if self.use_mlx:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            sampling_config = {
                "max_tokens": 128,
                "temperature": 0.8,
            }
            
            try:
                from src.utils_mlx import generate_mlx
                response = generate_mlx(self.model, self.tokenizer, [prompt], sampling_config)
                response_text = response[0].get("text", "").strip()
            except Exception as e:
                logging.error(f"MLX generation failed: {e}")
                return None
        else:
            # HuggingFace generation
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message}
            ]
            
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=128,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response_text = self.tokenizer.decode(
                    outputs[0][inputs.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
            except Exception as e:
                logging.error(f"HF generation failed: {e}")
                return None
        
        try:
            # Parse the JSON configuration
            config = json.loads(response_text)
            
            # Validate configuration structure
            if not self._validate_config(config):
                logging.warning(f"Invalid config generated: {config}")
                return None
                
            return {
                "config": config,
                "prompt": prompt if self.use_mlx else user_message,
                "response": response_text,
                "task_name": task.name
            }
            
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse JSON config: {e}")
            logging.debug(f"Raw response: {response_text}")
            return None
    
    def _format_task_examples(self, task: ARCTask) -> str:
        """Format task examples exactly like original SEAL."""
        train_examples = task.serialize()['train']
        formatted_examples = ""

        for example in train_examples:
            # Format input grid
            input_grid = example['input']
            input_str = "Input:\n"
            for row in input_grid:
                input_str += " ".join(map(str, row)) + "\n"
            
            # Format output grid
            output_grid = example['output']
            output_str = "\nOutput:\n"
            for row in output_grid:
                output_str += " ".join(map(str, row)) + "\n"
            
            formatted_examples += input_str + output_str + "\n"
        
        return formatted_examples
    
    def _validate_config(self, config: Dict) -> bool:
        """Validate that config has required structure."""
        try:
            # Check required keys
            if "data_generation" not in config or "training" not in config:
                return False
            
            data_gen = config["data_generation"]
            training = config["training"]
            
            # Check data generation keys
            required_data_keys = ["use_basic_augmentations", "use_size_augmentations", 
                                "use_chain_augmentations", "use_repeat_augmentations"]
            if not all(key in data_gen for key in required_data_keys):
                return False
            
            # Check training keys
            if not all(key in training for key in ["strategy", "learning_rate", "num_train_epochs"]):
                return False
            
            # Validate training strategy
            if training["strategy"] not in ["train_using_all_tokens", "train_using_output_tokens"]:
                return False
            
            # Validate numeric values
            if not isinstance(training["learning_rate"], (int, float)) or training["learning_rate"] <= 0:
                return False
            
            if not isinstance(training["num_train_epochs"], int) or training["num_train_epochs"] < 1:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _config_to_key(self, config: Dict) -> tuple:
        """Convert config to hashable key for duplicate checking."""
        data_gen = tuple(sorted(config["data_generation"].items()))
        training = tuple(sorted(config["training"].items()))
        return (data_gen, training)
    
    def generate_batch_self_edits(
        self, 
        tasks: List[ARCTask], 
        n_configs_per_task: int = 15
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate training configurations for a batch of ARC tasks.
        
        CORRECTED: Returns configurations, not solutions.
        
        Returns:
            Dictionary mapping task names to their generated configurations
        """
        results = {}
        
        for task in tasks:
            logging.info(f"Generating configurations for task {task.name}")
            
            try:
                configs = self.generate_self_edit_configs(task, n_configs_per_task)
                results[task.name] = configs
                
                logging.info(f"Generated {len(configs)} configurations for {task.name}")
                
            except Exception as e:
                logging.error(f"Failed to generate configs for task {task.name}: {e}")
                results[task.name] = []
        
        return results


def load_arc_tasks_from_json(challenge_file: str, solution_file: Optional[str] = None) -> List[ARCTask]:
    """
    Load ARC tasks from the standard ARC JSON format.
    """
    with open(challenge_file, 'r') as f:
        challenges = json.load(f)
    
    solutions = {}
    if solution_file:
        with open(solution_file, 'r') as f:
            solutions = json.load(f)
    
    tasks = []
    for task_id, task_data in challenges.items():
        # Parse training examples
        train_examples = []
        for example_data in task_data['train']:
            input_grid = np.array(example_data['input'])
            output_grid = np.array(example_data['output'])
            train_examples.append(ARCExample(input_grid, output_grid))
        
        # Parse test examples
        test_examples = []
        for i, example_data in enumerate(task_data['test']):
            input_grid = np.array(example_data['input'])
            
            # Try to get output from solutions file
            output_grid = None
            if task_id in solutions and i < len(solutions[task_id]):
                output_grid = np.array(solutions[task_id][i])
            elif 'output' in example_data:
                output_grid = np.array(example_data['output'])
            
            test_examples.append(ARCExample(input_grid, output_grid))
        
        task = ARCTask(
            name=task_id,
            train_examples=train_examples,
            test_examples=test_examples
        )
        tasks.append(task)
    
    return tasks


def main():
    """
    Example usage of ARCSelfEditGenerator
    """
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load ARC tasks (you'll need to provide the actual file paths)
    # tasks = load_arc_tasks_from_json("path/to/arc_challenges.json")
    
    # For testing, create a simple mock task
    mock_input = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mock_output = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    mock_test_input = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]])
    
    mock_task = ARCTask(
        name="test_task",
        train_examples=[ARCExample(mock_input, mock_output)],
        test_examples=[ARCExample(mock_test_input)]
    )
    
    # Initialize generator
    generator = ARCSelfEditGenerator("mlx-community/Meta-Llama-3-8B-Instruct")
    
    # Generate self-edits
    results = generator.generate_batch_self_edits([mock_task], n_configs_per_task=5)
    
    # Save results
    output_file = "arc_self_edit_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"ARC self-edit generation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
