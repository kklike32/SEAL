# knowledge-incorporation/src/restem/arc_restem_trainer.py

"""
ARC RestEM Training - Final Phase

This module implements Phase 4 of RestEM for ARC tasks:
- Take only the verified correct solutions
- Train final model using behavioral cloning (supervised learning)
- Creates the "RL_trained_model" that matches original SEAL paper
"""

import os
import json
import logging
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


class ARCRestEMTrainer:
    """
    RestEM training for ARC tasks.
    
    This implements the final phase of RestEM:
    1. Filter for only correct solutions from evaluation phase
    2. Train final model using behavioral cloning on correct solutions
    3. Create "RL_trained_model" matching original SEAL methodology
    """
    
    def __init__(self, model_name: str, lora_config: Optional[Dict] = None):
        self.model_name = model_name
        
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for RestEM training")
            
        self.base_model, self.tokenizer = mlx_lm_load(model_name)
        
        # RestEM uses lower rank than TTT phase
        self.lora_config = lora_config or {
            "rank": 16,   # Lower rank for final model
            "alpha": 16,
            "dropout": 0.0,
            "modules": ["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
        }
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        logging.info(f"Initialized ARCRestEMTrainer with {model_name}")
    
    def create_restem_iteration(
        self,
        self_edit_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, List[Dict[str, Any]]],
        output_dir: str,
        iteration_num: int = 1,
        epochs: int = 8,
        learning_rate: float = 5e-5
    ) -> str:
        """
        Create a RestEM model iteration by training on correct solutions only.
        
        Args:
            self_edit_results: Results from self-edit generation phase
            evaluation_results: Results from solution evaluation phase
            output_dir: Directory to save the trained model
            iteration_num: Iteration number for naming
            epochs: Training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Path to the saved RestEM model
        """
        logging.info(f"Creating RestEM iteration {iteration_num}")
        
        # Filter for correct solutions only
        correct_solutions = self._filter_correct_solutions(
            self_edit_results, evaluation_results
        )
        
        if not correct_solutions:
            logging.warning("No correct solutions found for RestEM training")
            return None
        
        logging.info(f"Found {len(correct_solutions)} correct solutions for training")
        
        # Create output directory for this iteration
        model_output_dir = os.path.join(output_dir, f"RL_trained_model_iteration_{iteration_num}")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Save correct solutions for reference
        correct_solutions_path = os.path.join(output_dir, "correct_solutions.json")
        with open(correct_solutions_path, 'w') as f:
            json.dump({
                "iteration": iteration_num,
                "num_correct_solutions": len(correct_solutions),
                "solutions": correct_solutions
            }, f, indent=2)
        
        # Create and train RestEM model
        model = self._create_restem_model()
        
        # Prepare training data
        training_data = self._prepare_restem_training_data(correct_solutions)
        
        # Train the model
        success = self._train_restem_model(
            model, 
            training_data, 
            model_output_dir,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        if success:
            logging.info(f"RestEM model saved to: {model_output_dir}")
            return model_output_dir
        else:
            logging.error("RestEM training failed")
            return None
    
    def _filter_correct_solutions(
        self,
        self_edit_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Filter solutions to keep only those marked as correct in evaluation.
        """
        correct_solutions = []
        
        for task_name, task_evaluations in evaluation_results.items():
            task_solutions = self_edit_results.get(task_name, {}).get("solutions", [])
            task_configs = self_edit_results.get(task_name, {}).get("configs", [])
            
            for i, eval_result in enumerate(task_evaluations):
                if eval_result.get("is_correct", False):
                    # This solution was marked as correct
                    solution_text = task_solutions[i] if i < len(task_solutions) else ""
                    config = task_configs[i] if i < len(task_configs) else {}
                    
                    if solution_text:
                        correct_solutions.append({
                            "task_name": task_name,
                            "solution_index": i,
                            "config": config,
                            "solution_text": solution_text,
                            "evaluation_result": eval_result
                        })
        
        return correct_solutions
    
    def _create_restem_model(self):
        """Create LoRA model for RestEM training."""
        model = linear_to_lora_layers(
            self.base_model,
            rank=self.lora_config["rank"],
            num_layers=16
        )
        return model
    
    def _prepare_restem_training_data(self, correct_solutions: List[Dict[str, Any]]) -> List[Dict]:
        """
        Prepare training data from correct solutions.
        
        This formats the solutions for behavioral cloning training.
        """
        training_data = []
        
        for solution_data in correct_solutions:
            solution_text = solution_data["solution_text"]
            
            # The solution should be in format:
            # <prompt><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<response>
            full_text = solution_text
            if not full_text.endswith("<|eot_id|>"):
                full_text += "<|eot_id|>"
            
            # Tokenize
            tokens = self.tokenizer.encode(full_text, return_tensors="pt")
            
            # Find assistant response start for label masking
            assistant_start = -1
            for i in range(len(tokens[0]) - 1):
                if tokens[0][i] == 128007 and tokens[0][i + 1] == 271:  # [start_header_id] assistant [end_header_id]
                    assistant_start = i + 2
            
            # Create labels - only learn from assistant response
            labels = tokens[0].clone()
            if assistant_start > 0:
                labels[:assistant_start] = -100
            
            training_data.append({
                "input_ids": tokens[0],
                "labels": labels,
                "task_name": solution_data["task_name"],
                "config": solution_data["config"]
            })
        
        return training_data
    
    def _train_restem_model(
        self,
        model,
        training_data: List[Dict],
        output_dir: str,
        epochs: int = 8,
        learning_rate: float = 5e-5,
        batch_size: int = 5  # Matching original SEAL batch size
    ) -> bool:
        """
        Train the RestEM model using behavioral cloning.
        """
        try:
            if not training_data:
                logging.error("No training data provided")
                return False
            
            logging.info(f"Training RestEM model with {len(training_data)} samples")
            
            # Initialize optimizer with cosine schedule (matching original)
            optimizer = mx.optimizers.AdamW(learning_rate=learning_rate)
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                
                # Shuffle data each epoch
                import random
                random.shuffle(training_data)
                
                for batch_start in range(0, len(training_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(training_data))
                    batch = training_data[batch_start:batch_end]
                    
                    # Prepare batch tensors
                    max_length = max(len(item["input_ids"]) for item in batch)
                    
                    batch_input_ids = []
                    batch_labels = []
                    
                    for item in batch:
                        input_ids = item["input_ids"]
                        labels = item["labels"]
                        
                        # Pad to max length
                        pad_length = max_length - len(input_ids)
                        if pad_length > 0:
                            input_ids = mx.concatenate([input_ids, mx.full((pad_length,), self.tokenizer.pad_token_id)])
                            labels = mx.concatenate([labels, mx.full((pad_length,), -100)])
                        
                        batch_input_ids.append(input_ids)
                        batch_labels.append(labels)
                    
                    input_ids = mx.stack(batch_input_ids)
                    labels = mx.stack(batch_labels)
                    
                    # Forward pass and loss
                    def loss_fn(model, input_ids, labels):
                        logits = model(input_ids)
                        # Cross-entropy loss only on non-masked tokens
                        valid_mask = (labels != -100)
                        if mx.sum(valid_mask) == 0:
                            return mx.array(0.0)
                        
                        # Compute loss
                        shifted_logits = logits[..., :-1, :]
                        shifted_labels = labels[..., 1:]
                        shifted_mask = valid_mask[..., 1:]
                        
                        loss = mx.where(
                            shifted_mask,
                            -mx.log(mx.softmax(shifted_logits)[mx.arange(shifted_labels.shape[0])[:, None], 
                                                              mx.arange(shifted_labels.shape[1])[None, :], 
                                                              shifted_labels]),
                            0.0
                        )
                        
                        return mx.sum(loss) / mx.sum(shifted_mask)
                    
                    # Compute gradients and update
                    loss, grads = mx.value_and_grad(loss_fn)(model, input_ids, labels)
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                logging.info(f"RestEM Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Save the trained model
            model.save_weights(os.path.join(output_dir, "adapter.safetensors"))
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training config
            config_path = os.path.join(output_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    "model_type": "RestEM",
                    "base_model": self.model_name,
                    "training_samples": len(training_data),
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "lora_config": self.lora_config
                }, f, indent=2)
            
            logging.info("RestEM training completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"RestEM training failed: {e}")
            return False


def main():
    """
    Example usage of ARCRestEMTrainer
    """
    logging.basicConfig(level=logging.INFO)
    
    # Mock data for testing
    mock_self_edit_results = {
        "test_task": {
            "configs": [
                {"config_id": 0, "strategy": "Look for color patterns"},
                {"config_id": 1, "strategy": "Find geometric transformations"}
            ],
            "solutions": [
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve this ARC task...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLooking at the pattern, I can see that colors are inverted...",
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve this ARC task...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe transformation is a 90-degree rotation..."
            ]
        }
    }
    
    mock_evaluation_results = {
        "test_task": [
            {"is_correct": True, "task_name": "test_task", "accuracy_score": 1.0},
            {"is_correct": False, "task_name": "test_task", "accuracy_score": 0.0}
        ]
    }
    
    # Initialize trainer
    trainer = ARCRestEMTrainer("mlx-community/Meta-Llama-3-8B-Instruct")
    
    # Create RestEM iteration
    model_path = trainer.create_restem_iteration(
        self_edit_results=mock_self_edit_results,
        evaluation_results=mock_evaluation_results,
        output_dir="test_restem_output",
        iteration_num=1,
        epochs=2  # Reduced for testing
    )
    
    logging.info(f"RestEM training complete. Model path: {model_path}")


if __name__ == "__main__":
    main()
