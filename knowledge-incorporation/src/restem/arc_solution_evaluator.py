# knowledge-incorporation/src/restem/arc_solution_evaluator.py

"""
ARC Configuration Evaluation for RestEM - CORRECTED VERSION

This module implements Phase 3 of RestEM for ARC tasks following original SEAL:
- Load each trained LoRA adapter (trained on augmented data per configuration)
- Test it on the original ARC problem to generate solutions
- Determine which configurations lead to correct solutions
- Assign binary rewards for RestEM behavioral cloning
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_lm_load
    from src.utils_mlx import generate_mlx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .arc_augmenters import ARCTask, ARCExample
# Import shared model manager
from .shared_model import get_shared_model


class ARCSolutionEvaluator:
    """
    Evaluates ARC configurations by testing trained adapters.
    
    CORRECTED: This implements Phase 3 of RestEM:
    1. Load each trained LoRA adapter (trained on config-based augmented data)
    2. Generate a solution for the original ARC test case using the adapter
    3. Compare against ground truth to determine if configuration was successful
    4. Assign binary reward (correct/incorrect) for behavioral cloning
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for solution evaluation")
            
        # Use shared model instead of loading our own
        self.base_model, self.tokenizer = get_shared_model(model_name)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # System message from original SEAL
        self.system_message = "You are a helpful assistant that provide the correct output for the given task immediately."
            
        logging.info(f"Initialized ARCSolutionEvaluator with {model_name}")
    
    def evaluate_config_adapter(
        self,
        task: ARCTask,
        adapter_path: str,
        config_data: Dict[str, Any],
        max_tokens: int = 200
    ) -> Dict[str, Any]:
        """
        Evaluate a configuration adapter on an ARC task.
        
        CORRECTED: Tests whether the adapter (trained on config-based augmented data)
        can solve the original task correctly.
        
        Args:
            task: The ARC task to evaluate on
            adapter_path: Path to the trained LoRA adapter
            config_data: Configuration used to train this adapter
            max_tokens: Maximum tokens for generation
            
        Returns:
            Dictionary with evaluation results including binary reward
        """
        try:
            logging.debug(f"Evaluating adapter {adapter_path} on task {task.name}")
            
            # Load the adapter
            adapter_model = self._load_adapter(adapter_path)
            if adapter_model is None:
                return {
                    "success": False,
                    "error": "Failed to load adapter",
                    "reward": 0,
                    "config": config_data["config"]
                }
            
            # Generate solution using the adapter
            solution_grid = self._generate_solution_with_adapter(
                adapter_model, 
                task, 
                max_tokens
            )
            
            if solution_grid is None:
                return {
                    "success": False,
                    "error": "Failed to generate solution",
                    "reward": 0,
                    "config": config_data["config"]
                }
            
            # Evaluate correctness against ground truth
            reward = self._evaluate_solution_correctness(task, solution_grid)
            
            # Clean up adapter model
            del adapter_model
            
            result = {
                "success": True,
                "reward": reward,
                "config": config_data["config"],
                "solution_grid": solution_grid.tolist() if solution_grid is not None else None,
                "correct": reward > 0
            }
            
            logging.debug(f"Evaluation result: reward={reward}")
            return result
            
        except Exception as e:
            logging.error(f"Failed to evaluate adapter {adapter_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "reward": 0,
                "config": config_data.get("config", {})
            }
    
    def _load_adapter(self, adapter_path: str):
        """Load a LoRA adapter from disk."""
        try:
            if not os.path.exists(adapter_path):
                logging.error(f"Adapter path does not exist: {adapter_path}")
                return None
            
            # Load adapter using MLX
            from mlx_lm.tuner import load_adapter
            adapter_model = load_adapter(adapter_path, self.base_model)
            
            return adapter_model
            
        except Exception as e:
            logging.error(f"Failed to load adapter from {adapter_path}: {e}")
            return None
    
    def _generate_solution_with_adapter(
        self, 
        adapter_model, 
        task: ARCTask, 
        max_tokens: int
    ) -> Optional[np.ndarray]:
        """Generate a solution using the adapter model."""
        try:
            # Create prompt for the test case
            test_example = task.test_examples[0] if task.test_examples else None
            if test_example is None:
                logging.error("No test example found in task")
                return None
            
            prompt = self._create_evaluation_prompt(task)
            
            # Generate solution
            sampling_config = {
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Low temperature for evaluation
                "top_p": 0.9
            }
            
            response = generate_mlx(adapter_model, self.tokenizer, [prompt], sampling_config)
            solution_text = response[0].get("text", "").strip()
            
            # Parse the solution text to extract grid
            solution_grid = self._parse_solution_grid(solution_text)
            
            return solution_grid
            
        except Exception as e:
            logging.error(f"Failed to generate solution: {e}")
            return None
    
    def _create_evaluation_prompt(self, task: ARCTask) -> str:
        """Create evaluation prompt following original SEAL format."""
        # Format training examples
        examples_text = ""
        for i, example in enumerate(task.train_examples):
            examples_text += f"\nExample {i+1}:\n"
            examples_text += f"Input:\n{self._format_grid(example.input_grid)}\n"
            examples_text += f"Output:\n{self._format_grid(example.output_grid)}\n"
        
        # Format test input
        test_example = task.test_examples[0]
        test_text = f"\nTest Input:\n{self._format_grid(test_example.input_grid)}\n"
        
        user_message = f"Here are the training examples:{examples_text}\n{test_text}\nWhat should the test output be?"
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def _format_grid(self, grid: np.ndarray) -> str:
        """Format a grid as string following original SEAL format."""
        rows = []
        for row in grid:
            rows.append(" ".join(str(cell) for cell in row))
        return "\n".join(rows)
    
    def _parse_solution_grid(self, solution_text: str) -> Optional[np.ndarray]:
        """Parse solution text to extract grid."""
        try:
            # Look for grid-like patterns in the text
            lines = solution_text.strip().split('\n')
            
            # Find lines that look like grid rows (numbers separated by spaces)
            grid_lines = []
            for line in lines:
                line = line.strip()
                if re.match(r'^[\d\s]+$', line) and len(line.split()) > 1:
                    grid_lines.append([int(x) for x in line.split()])
            
            if not grid_lines:
                return None
            
            # Check if all rows have the same length
            row_lengths = [len(row) for row in grid_lines]
            if len(set(row_lengths)) != 1:
                return None
            
            return np.array(grid_lines)
            
        except Exception as e:
            logging.debug(f"Failed to parse solution grid: {e}")
            return None
    
    def _evaluate_solution_correctness(self, task: ARCTask, solution_grid: np.ndarray) -> int:
        """Evaluate if solution is correct (binary reward)."""
        try:
            test_example = task.test_examples[0]
            if test_example.output_grid is None:
                # No ground truth available
                return 0
            
            ground_truth = test_example.output_grid
            
            # Check exact match
            if solution_grid.shape == ground_truth.shape and np.array_equal(solution_grid, ground_truth):
                return 1
            else:
                return 0
                
        except Exception as e:
            logging.debug(f"Failed to evaluate correctness: {e}")
            return 0
    
    def evaluate_batch_configs(
        self,
        task: ARCTask,
        adapter_paths: List[str],
        configs_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple configuration adapters.
        
        Returns evaluation results for behavioral cloning training data.
        """
        results = []
        
        for adapter_path, config_data in zip(adapter_paths, configs_data):
            if not os.path.exists(adapter_path):
                logging.warning(f"Adapter not found: {adapter_path}")
                continue
            
            result = self.evaluate_config_adapter(task, adapter_path, config_data)
            results.append(result)
        
        # Calculate statistics
        successful_evals = [r for r in results if r["success"]]
        correct_configs = [r for r in successful_evals if r["reward"] > 0]
        
        logging.info(f"Evaluated {len(successful_evals)}/{len(results)} configs successfully")
        logging.info(f"Found {len(correct_configs)} correct configurations")
        
        return results
    
    def create_behavioral_cloning_data(
        self,
        task: ARCTask,
        evaluation_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create behavioral cloning training data from evaluation results.
        
        This is used for Phase 4 (RestEM behavioral cloning).
        """
        bc_data = []
        
        for result in evaluation_results:
            if not result["success"]:
                continue
            
            # Create training example for behavioral cloning
            # Format: original task prompt -> configuration that worked
            config = result["config"]
            reward = result["reward"]
            
            # Format task as input
            task_prompt = self._format_task_for_bc(task)
            
            # Format configuration as target
            config_target = json.dumps(config, separators=(',', ':'))
            
            bc_example = {
                "input": task_prompt,
                "target": config_target,
                "reward": reward,
                "correct": reward > 0,
                "task_name": task.name
            }
            
            bc_data.append(bc_example)
        
        return bc_data
    
    def _format_task_for_bc(self, task: ARCTask) -> str:
        """Format task for behavioral cloning input."""
        examples_text = ""
        for i, example in enumerate(task.train_examples):
            examples_text += f"Input:\n{self._format_grid(example.input_grid)}\n"
            examples_text += f"Output:\n{self._format_grid(example.output_grid)}\n\n"
        
        return examples_text + "------\n\n" + """
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

Respond with a valid JSON object. Do not include any explanation, markdown, or extra text. Use lowercase `true`/`false` for booleans and ensure correct JSON syntax.
"""


def test_evaluator():
    """Test the solution evaluator."""
    logging.basicConfig(level=logging.INFO)
    
    # Create mock ARC task
    mock_input = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mock_output = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    
    task = ARCTask(
        name="test_task",
        train_examples=[ARCExample(mock_input, mock_output)],
        test_examples=[ARCExample(mock_input, mock_output)]  # With ground truth for testing
    )
    
    # Create mock config
    config_data = {
        "config": {
            "data_generation": {
                "use_basic_augmentations": True,
                "use_size_augmentations": False,
                "use_chain_augmentations": False,
                "use_repeat_augmentations": False
            },
            "training": {
                "strategy": "train_using_all_tokens",
                "learning_rate": 0.0001,
                "num_train_epochs": 3
            }
        }
    }
    
    # Test evaluation (would need actual adapter in real use)
    evaluator = ARCSolutionEvaluator("mlx-community/Meta-Llama-3-8B-Instruct")
    
    print("ARCSolutionEvaluator initialized successfully")
    print("Ready to evaluate trained adapters")


if __name__ == "__main__":
    test_evaluator()
