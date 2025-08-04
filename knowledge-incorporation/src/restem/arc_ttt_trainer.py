# knowledge-incorporation/src/restem/arc_ttt_trainer.py

"""
ARC Test-Time Training (TTT) for RestEM - CORRECTED VERSION

This module implements Phase 2 of RestEM for ARC tasks following the original SEAL:
- Train LoRA adapters on AUGMENTED TASK DATA based on configurations
- Each adapter is trained on augmented versions of the original task
- NO training on solution text - trains on task format following configurations
"""

import os
import json
import logging
import gc
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.tuner import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from .arc_augmenters import ARCAugmentationManager, ARCTask, ARCExample
# Import shared model manager
from .shared_model import get_shared_model


class ARCTTTTrainer:
    """
    Test-Time Training for ARC tasks based on configurations.
    
    CORRECTED: Trains LoRA adapters on augmented task data, not solution text.
    This follows the original SEAL methodology exactly.
    """
    
    def __init__(self, model_name: str, lora_config: Optional[Dict] = None):
        self.model_name = model_name
        
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for TTT training")
            
        # Use shared model instead of loading our own
        self.base_model, self.tokenizer = get_shared_model(model_name)
        
        # Default LoRA configuration matching original SEAL
        self.lora_config = lora_config or {
            "rank": 128,  # High rank like original
            "alpha": 16,
            "dropout": 0.0,
            "modules": ["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
        }
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.augmentation_manager = ARCAugmentationManager()
        
        # System message from original SEAL
        self.system_message = "You are a helpful assistant that provide the correct output for the given task immediately."
            
        logging.info(f"Initialized ARCTTTTrainer with {model_name}")
    
    def train_config_adapter(
        self,
        task: ARCTask,
        config_data: Dict[str, Any],
        output_dir: str,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: int = 1
    ) -> Optional[str]:
        """
        Train a LoRA adapter based on a configuration.
        
        CORRECTED: Trains on augmented task data, not solution text.
        This is the actual SEAL methodology.
        
        Args:
            task: Original ARC task
            config_data: Configuration dictionary specifying augmentation strategies
            output_dir: Directory to save the trained adapter
            epochs: Number of training epochs (from config if not specified)
            learning_rate: Learning rate (from config if not specified)
            batch_size: Training batch size
            
        Returns:
            Path to the saved adapter or None if training failed
        """
        try:
            config = config_data["config"]
            training_config = config["training"]
            
            # Use config parameters if not overridden
            if epochs is None:
                epochs = training_config.get("num_train_epochs", 3)
            if learning_rate is None:
                learning_rate = training_config.get("learning_rate", 5e-5)
            
            logging.info(f"Training adapter for task {task.name} with config")
            logging.info(f"Training params: epochs={epochs}, lr={learning_rate}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate augmented training data based on configuration
            augmented_tasks = self.augmentation_manager.apply_config_to_task(task, config)
            
            if not augmented_tasks:
                logging.warning("No augmented tasks generated, skipping adapter training")
                return None
            
            # Create training dataset from augmented tasks
            training_data = self._create_training_dataset(
                augmented_tasks, 
                training_config.get("strategy", "train_using_all_tokens")
            )
            
            if not training_data:
                logging.warning("No training data created, skipping adapter training")
                return None
            
            # Create LoRA model
            lora_model = self._create_lora_model()
            
            # Train the adapter
            adapter_path = self._train_adapter(
                lora_model,
                training_data,
                output_dir,
                epochs,
                learning_rate,
                batch_size
            )
            
            # Clean up memory
            del lora_model
            gc.collect()
            
            return adapter_path
            
        except Exception as e:
            logging.error(f"Failed to train config adapter: {e}")
            return None
    
    def _create_training_dataset(
        self, 
        augmented_tasks: List[ARCTask], 
        strategy: str
    ) -> List[Dict[str, Any]]:
        """
        Create training dataset from augmented tasks.
        
        CORRECTED: Creates input-output pairs for training, not solution text.
        """
        training_examples = []
        
        for aug_task in augmented_tasks:
            for train_ex in aug_task.train_examples:
                # Create input prompt
                input_text = self._format_arc_example_input(train_ex)
                
                # Create target output based on strategy
                if strategy == "train_using_all_tokens":
                    target_text = input_text + self._format_arc_example_output(train_ex)
                else:  # train_using_output_tokens
                    target_text = self._format_arc_example_output(train_ex)
                
                training_examples.append({
                    "input": input_text,
                    "target": target_text,
                    "task_name": aug_task.name
                })
        
        logging.info(f"Created {len(training_examples)} training examples")
        return training_examples
    
    def _format_arc_example_input(self, example: ARCExample) -> str:
        """Format ARC example input following original SEAL format."""
        input_str = "Input:\n"
        for row in example.input_grid:
            input_str += " ".join(map(str, row)) + "\n"
        
        input_str += "\nOutput:\n"
        return input_str
    
    def _format_arc_example_output(self, example: ARCExample) -> str:
        """Format ARC example output following original SEAL format."""
        if example.output_grid is None:
            return ""
        
        output_str = ""
        for row in example.output_grid:
            output_str += " ".join(map(str, row)) + "\n"
        
        return output_str.strip()
    
    def _create_lora_model(self):
        """Create LoRA model from base model."""
        try:
            # Convert linear layers to LoRA
            lora_model = linear_to_lora_layers(
                self.base_model,
                num_lora_layers=self.lora_config["rank"],
                config={
                    "lora_layers": self.lora_config["modules"],
                    "lora_rank": self.lora_config["rank"], 
                    "lora_alpha": self.lora_config["alpha"],
                    "lora_dropout": self.lora_config["dropout"]
                }
            )
            
            return lora_model
            
        except Exception as e:
            logging.error(f"Failed to create LoRA model: {e}")
            raise
    
    def _train_adapter(
        self,
        model,
        training_data: List[Dict[str, Any]],
        output_dir: str,
        epochs: int,
        learning_rate: float,
        batch_size: int
    ) -> str:
        """Train the LoRA adapter."""
        try:
            # Prepare training arguments
            training_args = {
                "num_epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "save_every": epochs,  # Save at the end
                "output_dir": output_dir
            }
            
            # Convert training data to MLX format
            mlx_dataset = self._prepare_mlx_dataset(training_data)
            
            # Train using MLX-LM tuner
            from mlx_lm.tuner import train
            
            train(
                model=model,
                tokenizer=self.tokenizer,
                dataset=mlx_dataset,
                **training_args
            )
            
            adapter_path = os.path.join(output_dir, "adapters.npz")
            
            if os.path.exists(adapter_path):
                logging.info(f"Adapter saved to {adapter_path}")
                return adapter_path
            else:
                logging.error("Adapter file not found after training")
                return None
                
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return None
    
    def _prepare_mlx_dataset(self, training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare dataset for MLX training."""
        mlx_data = []
        
        for example in training_data:
            # Tokenize input and target
            input_ids = self.tokenizer.encode(example["input"], add_special_tokens=False)
            target_ids = self.tokenizer.encode(example["target"], add_special_tokens=False)
            
            # Combine for causal LM training
            full_ids = input_ids + target_ids
            
            # Create labels (same as input_ids for causal LM)
            labels = full_ids.copy()
            
            # Mask input portion if using output-only strategy
            if "train_using_output_tokens" in example.get("strategy", ""):
                labels[:len(input_ids)] = [-100] * len(input_ids)  # Ignore input tokens in loss
            
            mlx_data.append({
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * len(full_ids)
            })
        
        return mlx_data
    
    def train_batch_configs(
        self,
        task: ARCTask,
        configs_data: List[Dict[str, Any]],
        output_base_dir: str
    ) -> List[str]:
        """
        Train adapters for multiple configurations.
        
        Returns list of paths to successfully trained adapters.
        """
        successful_adapters = []
        
        for i, config_data in enumerate(configs_data):
            adapter_dir = os.path.join(output_base_dir, f"adapter_{i}")
            
            adapter_path = self.train_config_adapter(
                task, 
                config_data, 
                adapter_dir
            )
            
            if adapter_path:
                successful_adapters.append(adapter_path)
                
                # Save configuration info
                config_path = os.path.join(adapter_dir, "config.json")
                with open(config_path, 'w') as f:
                    json.dump({
                        "config_id": i,
                        "config": config_data["config"],
                        "task_name": task.name,
                        "adapter_path": adapter_path
                    }, f, indent=2)
            
            # Clean up between trainings
            gc.collect()
        
        logging.info(f"Successfully trained {len(successful_adapters)}/{len(configs_data)} adapters")
        return successful_adapters
    
    def _create_lora_model(self):
        """Create a LoRA-adapted version of the base model."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for LoRA model creation")
        
        # Convert linear layers to LoRA layers
        model = linear_to_lora_layers(
            self.base_model,
            rank=self.lora_config["rank"],
            num_layers=16,  # Adapt 16 layers like original
        )
        
        return model
    
    def _create_training_dataset(self, texts: List[str]) -> List[Dict]:
        """Create training dataset from text samples."""
        if not texts:
            return []
        
        dataset = []
        for text in texts:
            # Tokenize the text
            tokens = self.tokenizer.encode(text, return_tensors="pt")
            
            # Find the assistant response start (last occurrence of [128007, 271])
            assistant_start = -1
            for i in range(len(tokens[0]) - 1):
                if tokens[0][i] == 128007 and tokens[0][i + 1] == 271:
                    assistant_start = i + 2
            
            # Create labels - mask everything before assistant response
            labels = tokens[0].clone()
            if assistant_start > 0:
                labels[:assistant_start] = -100
            
            dataset.append({
                "input_ids": tokens[0],
                "labels": labels
            })
        
        return dataset
    
    def _train_adapter(
        self, 
        model, 
        dataset: List[Dict], 
        output_dir: str,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 1
    ) -> bool:
        """
        Train the LoRA adapter using MLX.
        
        This implements a simplified training loop similar to the original SEAL.
        """
        try:
            if not dataset:
                logging.warning("Empty dataset provided for training")
                return False
            
            # Initialize optimizer
            optimizer = mx.optimizers.AdamW(learning_rate=learning_rate)
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                
                for batch_start in range(0, len(dataset), batch_size):
                    batch_end = min(batch_start + batch_size, len(dataset))
                    batch = dataset[batch_start:batch_end]
                    
                    # Prepare batch data
                    input_ids = mx.array([item["input_ids"].numpy() for item in batch])
                    labels = mx.array([item["labels"].numpy() for item in batch])
                    
                    # Forward pass and loss computation
                    def loss_fn(model, input_ids, labels):
                        logits = model(input_ids)
                        # Simple cross-entropy loss
                        loss = mx.mean(mx.where(
                            labels != -100,
                            -mx.log_softmax(logits)[..., :-1] * mx.one_hot(labels[..., 1:], logits.shape[-1]),
                            0.0
                        ))
                        return loss
                    
                    # Compute gradients and update
                    loss, grads = mx.value_and_grad(loss_fn)(model, input_ids, labels)
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                logging.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Save the model
            model.save_weights(os.path.join(output_dir, "adapter.safetensors"))
            
            return True
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return False
    
    def train_batch_adapters(
        self,
        solutions_data: Dict[str, Dict[str, Any]],
        base_output_dir: str,
        epochs: int = 3
    ) -> Dict[str, List[Optional[str]]]:
        """
        Train adapters for a batch of solutions.
        
        Args:
            solutions_data: Dictionary mapping task names to their solution data
            base_output_dir: Base directory for saving adapters
            epochs: Training epochs per adapter
            
        Returns:
            Dictionary mapping task names to lists of adapter paths
        """
        adapter_paths = {}
        
        for task_name, task_data in solutions_data.items():
            logging.info(f"Training adapters for task {task_name}")
            
            task_adapter_paths = []
            configs = task_data.get("configs", [])
            solutions = task_data.get("solutions", [])
            
            for i, (config, solution) in enumerate(zip(configs, solutions)):
                # Create solution data
                solution_data = {
                    "config": config,
                    "solution": solution,
                    "config_id": i
                }
                
                # Create adapter output directory
                adapter_dir = os.path.join(base_output_dir, task_name, f"adapter_{i}")
                
                # Train adapter
                adapter_path = self.train_solution_adapter(
                    task_name=task_name,
                    solution_data=solution_data,
                    output_dir=adapter_dir,
                    epochs=epochs
                )
                
                task_adapter_paths.append(adapter_path)
            
            adapter_paths[task_name] = task_adapter_paths
            logging.info(f"Trained {len([p for p in task_adapter_paths if p])} adapters for task {task_name}")
        
        return adapter_paths


def main():
    """
    Example usage of ARCTTTTrainer
    """
    logging.basicConfig(level=logging.INFO)
    
    # Mock solution data for testing
    mock_solutions_data = {
        "test_task": {
            "configs": [
                {"config_id": 0, "strategy": "Look for color patterns", "temperature": 0.7},
                {"config_id": 1, "strategy": "Find geometric transformations", "temperature": 0.8}
            ],
            "solutions": [
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve this ARC task...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLooking at the pattern, I can see that colors are inverted in each example...",
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve this ARC task...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe transformation appears to be a 90-degree rotation..."
            ]
        }
    }
    
    # Initialize trainer
    trainer = ARCTTTTrainer("mlx-community/Meta-Llama-3-8B-Instruct")
    
    # Train adapters
    adapter_paths = trainer.train_batch_adapters(
        solutions_data=mock_solutions_data,
        base_output_dir="test_adapters",
        epochs=1  # Reduced for testing
    )
    
    logging.info(f"TTT training complete. Adapter paths: {adapter_paths}")


if __name__ == "__main__":
    main()
