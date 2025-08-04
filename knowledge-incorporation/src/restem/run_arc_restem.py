#!/usr/bin/env python3
"""
Run ARC RestEM Pipeline

This script runs the corrected SEAL RestEM implementation on ARC tasks.
Includes comprehensive logging and error handling.
"""

# Handle imports properly regardless of where script is run from
import sys
import os
import json
import logging
import argparse
from datetime import datetime
import numpy as np

# Get the script's directory and add proper paths
script_dir = os.path.dirname(os.path.abspath(__file__))
restem_dir = script_dir  # We're in the restem directory
src_dir = os.path.dirname(restem_dir)  # Go up to src
ki_dir = os.path.dirname(src_dir)  # Go up to knowledge-incorporation

# Add paths for imports
sys.path.insert(0, ki_dir)
sys.path.insert(0, restem_dir)

# Import modules
try:
    # Try relative imports first (if run from restem directory)
    from arc_restem_pipeline import ARCRestEMPipeline
    from arc_augmenters import ARCTask, ARCExample
except ImportError:
    try:
        # Try absolute imports (if run from SEAL root or elsewhere)
        from src.restem.arc_restem_pipeline import ARCRestEMPipeline
        from src.restem.arc_augmenters import ARCTask, ARCExample
    except ImportError as e:
        print(f"Import error: {e}")
        print(f"Script directory: {script_dir}")
        print(f"Knowledge-incorporation path: {ki_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        sys.exit(1)


def setup_logging(log_dir: str, log_level: str = "INFO") -> None:
    """Setup comprehensive logging."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"arc_restem_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def load_arc_data(data_dir: str, max_tasks: int = None) -> list:
    """Load ARC tasks from data directory."""
    challenges_file = os.path.join(data_dir, "arc-agi_training_challenges.json")
    solutions_file = os.path.join(data_dir, "arc-agi_training_solutions.json")
    
    if not os.path.exists(challenges_file):
        logging.warning(f"ARC challenges file not found: {challenges_file}")
        logging.info("Creating mock ARC tasks for testing...")
        return create_mock_tasks()
    
    try:
        # Load challenges
        with open(challenges_file, 'r') as f:
            challenges = json.load(f)
        
        # Load solutions
        with open(solutions_file, 'r') as f:
            solutions = json.load(f)
        
        tasks = []
        task_ids = list(challenges.keys())
        if max_tasks:
            task_ids = task_ids[:max_tasks]
        
        for task_id in task_ids:
            if task_id not in challenges or task_id not in solutions:
                continue
            
            challenge = challenges[task_id]
            solution = solutions[task_id]
            
            # Create training examples
            train_examples = []
            for train_ex in challenge['train']:
                train_examples.append(ARCExample(
                    input_grid=np.array(train_ex['input']),
                    output_grid=np.array(train_ex['output'])
                ))
            
            # Create test examples
            test_examples = []
            for i, test_ex in enumerate(challenge['test']):
                output_grid = None
                if i < len(solution):
                    output_grid = np.array(solution[i])
                
                test_examples.append(ARCExample(
                    input_grid=np.array(test_ex['input']),
                    output_grid=output_grid
                ))
            
            tasks.append(ARCTask(task_id, train_examples, test_examples))
        
        logging.info(f"Loaded {len(tasks)} ARC tasks from {data_dir}")
        return tasks
        
    except Exception as e:
        logging.error(f"Failed to load ARC data: {e}")
        logging.info("Creating mock ARC tasks for testing...")
        return create_mock_tasks()


def create_mock_tasks() -> list:
    """Create mock ARC tasks for testing."""
    mock_tasks = []
    
    # Task 1: Simple inversion pattern
    input1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    output1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    
    mock_tasks.append(ARCTask(
        name="mock_inversion",
        train_examples=[ARCExample(input1, output1)],
        test_examples=[ARCExample(input1, output1)]
    ))
    
    # Task 2: Color pattern
    input2 = np.array([[0, 2, 0], [2, 1, 2], [0, 2, 0]])
    output2 = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]])
    
    mock_tasks.append(ARCTask(
        name="mock_color_pattern",
        train_examples=[ARCExample(input2, output2)],
        test_examples=[ARCExample(input2, output2)]
    ))
    
    logging.info(f"Created {len(mock_tasks)} mock ARC tasks")
    return mock_tasks


def main():
    parser = argparse.ArgumentParser(description="Run ARC RestEM Pipeline")
    
    # Get SEAL root directory (3 levels up from restem)
    seal_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Default paths relative to SEAL root
    default_data_dir = os.path.join(seal_root, "few-shot", "data")
    default_output_dir = os.path.join(seal_root, "logs", "arc_restem_experiments")
    
    parser.add_argument("--data-dir", type=str, default=default_data_dir, 
                       help="Directory containing ARC data files")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                       help="Output directory for results and logs")
    parser.add_argument("--model-name", type=str, default="mlx-community/Meta-Llama-3-8B-Instruct",
                       help="Model name to use")
    parser.add_argument("--max-tasks", type=int, default=3,
                       help="Maximum number of tasks to process")
    parser.add_argument("--n-configs", type=int, default=15,
                       help="Number of configurations per task")
    parser.add_argument("--restem-epochs", type=int, default=8,
                       help="Number of epochs for RestEM training")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Configure MLX memory settings for Apple Silicon
    os.environ['MLX_GPU_MEMORY_LIMIT'] = '0.7'  # Use 70% of GPU memory
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['MLX_ENABLE_UNIFIED_MEMORY'] = '1'
    
    # Debug: Show detected paths
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Detected SEAL root: {seal_root}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir, args.log_level)
    
    try:
        logging.info("=" * 80)
        logging.info("ARC RestEM Pipeline - SEAL Methodology Implementation")
        logging.info("=" * 80)
        logging.info(f"Model: {args.model_name}")
        logging.info(f"Data directory: {args.data_dir}")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"Max tasks: {args.max_tasks}")
        logging.info(f"Configurations per task: {args.n_configs}")
        logging.info(f"RestEM epochs: {args.restem_epochs}")
        
        # Load ARC tasks
        logging.info("Loading ARC tasks...")
        arc_tasks = load_arc_data(args.data_dir, args.max_tasks)
        
        if not arc_tasks:
            logging.error("No ARC tasks loaded. Exiting.")
            return 1
        
        # Initialize pipeline
        logging.info("Initializing ARC RestEM Pipeline...")
        pipeline = ARCRestEMPipeline(
            model_name=args.model_name,
            output_base_dir=args.output_dir
        )
        
        # Run RestEM iteration
        iteration_name = f"arc_restem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logging.info("Starting RestEM iteration...")
        results = pipeline.run_complete_restem_iteration(
            arc_tasks=arc_tasks,
            iteration_name=iteration_name,
            n_configs=args.n_configs,
            restem_epochs=args.restem_epochs
        )
        
        # Log final results
        logging.info("=" * 80)
        logging.info("RestEM Iteration Complete!")
        logging.info("=" * 80)
        
        if "summary" in results:
            summary = results["summary"]
            logging.info(f"Total configurations: {summary['total_configurations']}")
            logging.info(f"Successful configurations: {summary['successful_configurations']}")
            logging.info(f"Success rate: {summary['success_rate']:.2%}")
            logging.info(f"Results directory: {summary['iteration_dir']}")
        
        if "error" in results:
            logging.error(f"Error occurred: {results['error']}")
            return 1
        
        logging.info(f"Log file: {log_file}")
        logging.info("Run completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logging.warning("Run interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
