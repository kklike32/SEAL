# knowledge-incorporation/src/restem/arc_restem_pipeline.py

"""
ARC RestEM Pipeline Orchestrator - CORRECTED VERSION

This module orchestrates the complete RestEM methodology for ARC tasks following 
the original SEAL exactly:
1. Generate training configurations (not solutions)
2. Train LoRA adapters on augmented data per configuration 
3. Evaluate configurations by testing adapters
4. RestEM behavioral cloning on successful configurations

CORRECTED: This implements the actual SEAL methodology - configuration-based.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from .arc_self_edit_generator import ARCSelfEditGenerator
from .arc_ttt_trainer import ARCTTTTrainer  
from .arc_solution_evaluator import ARCSolutionEvaluator
from .arc_restem_trainer import ARCRestEMTrainer
from .arc_augmenters import ARCTask, ARCExample


def load_arc_tasks_from_json(json_path: str) -> List[ARCTask]:
    """Load ARC tasks from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tasks = []
    for task_id, task_data in data.items():
        train_examples = []
        for ex in task_data.get('train', []):
            train_examples.append(ARCExample(
                input_grid=np.array(ex['input']),
                output_grid=np.array(ex['output'])
            ))
        
        test_examples = []
        for ex in task_data.get('test', []):
            output_grid = np.array(ex['output']) if 'output' in ex else None
            test_examples.append(ARCExample(
                input_grid=np.array(ex['input']),
                output_grid=output_grid
            ))
        
        tasks.append(ARCTask(task_id, train_examples, test_examples))
    
    return tasks


class ARCRestEMPipeline:
    """
    Complete ARC RestEM pipeline following the original SEAL methodology.
    
    CORRECTED: This orchestrates the configuration-based approach:
    1. Generate training configurations per ARC task (not solutions)
    2. Train LoRA adapters on augmented data based on configurations
    3. Evaluate configurations by testing their trained adapters
    4. Train final model on successful configurations only (behavioral cloning)
    """
    
    def __init__(
        self,
        model_name: str = "mlx-community/Meta-Llama-3-8B-Instruct",
        output_base_dir: str = "logs"
    ):
        self.model_name = model_name
        self.output_base_dir = output_base_dir
        
        # Initialize components
        self.self_edit_generator = ARCSelfEditGenerator(model_name, use_mlx=True)
        self.ttt_trainer = ARCTTTTrainer(model_name)
        self.solution_evaluator = ARCSolutionEvaluator(model_name)
        self.restem_trainer = ARCRestEMTrainer(model_name)
        
        logging.info(f"Initialized ARCRestEMPipeline with {model_name}")
    
    def run_complete_restem_iteration(
        self,
        arc_tasks: List[ARCTask],
        iteration_name: str = "restem_iteration_1",
        n_configs: int = 15,  # Number of configurations per task (original SEAL)
        restem_epochs: int = 8
    ) -> Dict[str, Any]:
        """
        Run a complete RestEM iteration on ARC tasks following original SEAL.
        
        CORRECTED: This implements the configuration-based approach:
        1. Generate training configurations (JSON objects)
        2. Train adapters on augmented data per configuration
        3. Evaluate configurations by testing adapters
        4. Behavioral cloning on successful configurations
        
        Args:
            arc_tasks: List of ARC tasks to process
            iteration_name: Name for this iteration
            n_configs: Number of configurations per ARC task
            restem_epochs: Epochs for final RestEM training
            
        Returns:
            Dictionary with results and paths
        """
        iteration_dir = os.path.join(self.output_base_dir, iteration_name)
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Setup logging for this iteration
        self._setup_iteration_logging(iteration_dir)
        
        logging.info(f"Starting RestEM iteration: {iteration_name}")
        logging.info(f"Processing {len(arc_tasks)} ARC tasks")
        
        results = {
            "iteration_name": iteration_name,
            "timestamp": datetime.now().isoformat(),
            "num_tasks": len(arc_tasks),
            "n_configs_per_task": n_configs,
            "phase_results": {}
        }
        
        try:
            # Phase 1: Generate training configurations (not solutions)
            logging.info("Phase 1: Generating training configurations")
            configs_dir = os.path.join(iteration_dir, "phase1_configs")
            task_configs = self._phase1_generate_configs(arc_tasks, configs_dir, n_configs)
            results["phase_results"]["phase1"] = {
                "configs_generated": sum(len(configs) for configs in task_configs.values()),
                "configs_dir": configs_dir
            }
            
            # Phase 2: Train adapters on augmented data per configuration
            logging.info("Phase 2: Training adapters on configuration-based augmented data")
            adapters_dir = os.path.join(iteration_dir, "phase2_adapters")
            adapter_paths = self._phase2_train_config_adapters(arc_tasks, task_configs, adapters_dir)
            results["phase_results"]["phase2"] = {
                "adapters_trained": sum(len([p for p in paths if p]) for paths in adapter_paths.values()),
                "adapters_dir": adapters_dir
            }
            
            # Phase 3: Evaluate configurations by testing adapters
            logging.info("Phase 3: Evaluating configurations")
            evaluations_dir = os.path.join(iteration_dir, "phase3_evaluations")
            evaluation_results = self._phase3_evaluate_configs(arc_tasks, task_configs, adapter_paths, evaluations_dir)
            results["phase_results"]["phase3"] = {
                "evaluations_completed": len(evaluation_results),
                "evaluations_dir": evaluations_dir
            }
            
            # Phase 4: RestEM behavioral cloning on successful configurations
            logging.info("Phase 4: RestEM behavioral cloning training")
            restem_dir = os.path.join(iteration_dir, "phase4_restem")
            restem_results = self._phase4_restem_training(arc_tasks, evaluation_results, restem_dir, restem_epochs)
            results["phase_results"]["phase4"] = restem_results
            
            # Final statistics
            total_configs = sum(len(configs) for configs in task_configs.values())
            successful_configs = sum(
                sum(1 for eval_result in task_evals if eval_result.get("reward", 0) > 0)
                for task_evals in evaluation_results.values()
            )
            
            results["summary"] = {
                "total_configurations": total_configs,
                "successful_configurations": successful_configs,
                "success_rate": successful_configs / total_configs if total_configs > 0 else 0,
                "iteration_dir": iteration_dir
            }
            
            logging.info(f"RestEM iteration complete: {successful_configs}/{total_configs} configs successful")
            
            # Save results
            results_path = os.path.join(iteration_dir, "iteration_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            logging.error(f"RestEM iteration failed: {e}")
            results["error"] = str(e)
            return results
    
    def _phase1_generate_configs(
        self, 
        arc_tasks: List[ARCTask], 
        configs_dir: str, 
        n_configs: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Phase 1: Generate training configurations for each task.
        
        CORRECTED: Generates JSON configurations, not solution text.
        """
        os.makedirs(configs_dir, exist_ok=True)
        task_configs = {}
        
        for task in arc_tasks:
            logging.info(f"Generating {n_configs} configurations for task {task.name}")
            
            try:
                configs = self.self_edit_generator.generate_self_edit_configs(task, n_configs)
                task_configs[task.name] = configs
                
                # Save configurations
                task_config_path = os.path.join(configs_dir, f"{task.name}_configs.json")
                with open(task_config_path, 'w') as f:
                    json.dump(configs, f, indent=2)
                
                logging.info(f"Generated {len(configs)} configurations for {task.name}")
                
            except Exception as e:
                logging.error(f"Failed to generate configs for {task.name}: {e}")
                task_configs[task.name] = []
        
        return task_configs
    
    def _phase2_train_config_adapters(
        self,
        arc_tasks: List[ARCTask],
        task_configs: Dict[str, List[Dict[str, Any]]],
        adapters_dir: str
    ) -> Dict[str, List[Optional[str]]]:
        """
        Phase 2: Train LoRA adapters based on configurations.
        
        CORRECTED: Trains on augmented data per configuration, not solution text.
        """
        os.makedirs(adapters_dir, exist_ok=True)
        task_lookup = {task.name: task for task in arc_tasks}
        adapter_paths = {}
        
        for task_name, configs in task_configs.items():
            if task_name not in task_lookup:
                continue
                
            task = task_lookup[task_name]
            task_adapter_dir = os.path.join(adapters_dir, task_name)
            
            logging.info(f"Training {len(configs)} adapters for task {task_name}")
            
            # Train adapters for this task's configurations
            task_adapter_paths = self.ttt_trainer.train_batch_configs(
                task, configs, task_adapter_dir
            )
            
            adapter_paths[task_name] = task_adapter_paths
            
            logging.info(f"Trained {len([p for p in task_adapter_paths if p])}/{len(configs)} adapters for {task_name}")
        
        return adapter_paths
    
    def _phase3_evaluate_configs(
        self,
        arc_tasks: List[ARCTask],
        task_configs: Dict[str, List[Dict[str, Any]]],
        adapter_paths: Dict[str, List[Optional[str]]],
        evaluations_dir: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Phase 3: Evaluate configurations by testing their trained adapters.
        """
        os.makedirs(evaluations_dir, exist_ok=True)
        task_lookup = {task.name: task for task in arc_tasks}
        evaluation_results = {}
        
        for task_name in task_configs.keys():
            if task_name not in task_lookup or task_name not in adapter_paths:
                continue
            
            task = task_lookup[task_name]
            configs = task_configs[task_name]
            adapters = adapter_paths[task_name]
            
            logging.info(f"Evaluating {len(configs)} configurations for task {task_name}")
            
            # Evaluate configurations
            task_evaluations = self.solution_evaluator.evaluate_batch_configs(
                task, adapters, configs
            )
            
            evaluation_results[task_name] = task_evaluations
            
            # Save evaluation results
            eval_path = os.path.join(evaluations_dir, f"{task_name}_evaluations.json")
            with open(eval_path, 'w') as f:
                json.dump(task_evaluations, f, indent=2)
            
            # Log summary
            successful = sum(1 for eval_result in task_evaluations if eval_result.get("reward", 0) > 0)
            logging.info(f"Task {task_name}: {successful}/{len(task_evaluations)} configurations successful")
        
        return evaluation_results
    
    def _phase4_restem_training(
        self,
        arc_tasks: List[ARCTask],
        evaluation_results: Dict[str, List[Dict[str, Any]]],
        restem_dir: str,
        epochs: int
    ) -> Dict[str, Any]:
        """
        Phase 4: RestEM behavioral cloning on successful configurations.
        """
        os.makedirs(restem_dir, exist_ok=True)
        
        # Create behavioral cloning dataset from evaluation results
        task_lookup = {task.name: task for task in arc_tasks if task.name in evaluation_results}
        
        bc_data = []
        for task_name, task_evaluations in evaluation_results.items():
            if task_name in task_lookup:
                task_bc_data = self.solution_evaluator.create_behavioral_cloning_data(
                    task_lookup[task_name], task_evaluations
                )
                bc_data.extend(task_bc_data)
        
        if not bc_data:
            logging.warning("No successful configurations found for RestEM training")
            return {"success": False, "error": "No training data"}
        
        # Filter for successful configurations only
        successful_data = [example for example in bc_data if example["reward"] > 0]
        
        logging.info(f"Training RestEM on {len(successful_data)} successful configurations")
        
        # Train final RestEM model
        restem_results = self.restem_trainer.train_restem_model(
            successful_data, restem_dir, epochs
        )
        
        return restem_results
    
    def _setup_iteration_logging(self, iteration_dir: str):
        """Setup logging for this iteration."""
        log_file = os.path.join(iteration_dir, "iteration.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)


def main():
    """Example usage of ARCRestEMPipeline."""
    logging.basicConfig(level=logging.INFO)
    
    # Create mock ARC tasks
    import numpy as np
    
    mock_input = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mock_output = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    
    mock_tasks = [
        ARCTask(
            name="test_task_1",
            train_examples=[ARCExample(mock_input, mock_output)],
            test_examples=[ARCExample(mock_input, mock_output)]
        ),
        ARCTask(
            name="test_task_2", 
            train_examples=[ARCExample(mock_input * 2, mock_output * 2)],
            test_examples=[ARCExample(mock_input * 2, mock_output * 2)]
        )
    ]
    
    # Initialize pipeline
    pipeline = ARCRestEMPipeline(
        model_name="mlx-community/Meta-Llama-3-8B-Instruct",
        output_base_dir="test_logs"
    )
    
    # Run RestEM iteration
    results = pipeline.run_complete_restem_iteration(
        arc_tasks=mock_tasks,
        iteration_name="test_iteration",
        n_configs=5,  # Smaller for testing
        restem_epochs=2
    )
    
    print(f"RestEM iteration results: {results}")


if __name__ == "__main__":
    main()
