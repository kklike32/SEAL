# knowledge-incorporation/src/restem/arc_augmenters.py

"""
ARC Augmentation Framework for SEAL RestEM Implementation

This implements the sophisticated augmentation system from the original SEAL methodology:
1. Basic augmentations (rotation, reflection, color permutation)
2. Size augmentations (cropping, padding)
3. Chain augmentations (sequences of transformations)
4. Repeat augmentations (multiple applications)
"""

import numpy as np
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools


@dataclass
class ARCExample:
    """Represents an ARC input-output example."""
    input_grid: np.ndarray
    output_grid: np.ndarray


@dataclass 
class ARCTask:
    """Represents an ARC task with training and test examples."""
    name: str
    train_examples: List[ARCExample]
    test_examples: List[ARCExample]
    
    def serialize(self) -> Dict:
        """Serialize task to match original format."""
        return {
            "train": [
                {"input": ex.input_grid.tolist(), "output": ex.output_grid.tolist()}
                for ex in self.train_examples
            ],
            "test": [
                {"input": ex.input_grid.tolist(), "output": ex.output_grid.tolist()}
                for ex in self.test_examples
            ]
        }


class ARCAugmenter(ABC):
    """Base class for ARC augmentations."""
    
    @abstractmethod
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        """Apply augmentation to a task."""
        pass
    
    def check_size_constraints(self, task: ARCTask, max_size: int = 30) -> bool:
        """Check that task meets size constraints."""
        for ex in task.train_examples + task.test_examples:
            if ex.input_grid.shape[0] > max_size or ex.input_grid.shape[1] > max_size:
                return False
            if ex.output_grid is not None and (ex.output_grid.shape[0] > max_size or ex.output_grid.shape[1] > max_size):
                return False
        return True


class RotateAugmenter(ARCAugmenter):
    """Rotate grids by specified degrees."""
    
    def __init__(self, degrees: int):
        self.degrees = degrees
        self.k = degrees // 90  # Number of 90-degree rotations
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        new_train = []
        for ex in task.train_examples:
            new_input = np.rot90(ex.input_grid, self.k)
            new_output = np.rot90(ex.output_grid, self.k) if ex.output_grid is not None else None
            new_train.append(ARCExample(new_input, new_output))
        
        new_test = []
        for ex in task.test_examples:
            new_input = np.rot90(ex.input_grid, self.k)
            new_output = np.rot90(ex.output_grid, self.k) if ex.output_grid is not None else None
            new_test.append(ARCExample(new_input, new_output))
        
        return ARCTask(task.name + f"_rot{self.degrees}", new_train, new_test)


class TransposeAugmenter(ARCAugmenter):
    """Transpose grids (swap rows and columns)."""
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        new_train = []
        for ex in task.train_examples:
            new_input = np.transpose(ex.input_grid)
            new_output = np.transpose(ex.output_grid) if ex.output_grid is not None else None
            new_train.append(ARCExample(new_input, new_output))
        
        new_test = []
        for ex in task.test_examples:
            new_input = np.transpose(ex.input_grid)
            new_output = np.transpose(ex.output_grid) if ex.output_grid is not None else None
            new_test.append(ARCExample(new_input, new_output))
        
        return ARCTask(task.name + "_transposed", new_train, new_test)


class IncreaseResolutionAugmenter(ARCAugmenter):
    """Increase resolution by scaling up grids."""
    
    def __init__(self, scale_factor: int = 2):
        self.scale_factor = scale_factor
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        new_train = []
        for ex in task.train_examples:
            new_input = self._scale_grid(ex.input_grid)
            new_output = self._scale_grid(ex.output_grid) if ex.output_grid is not None else None
            new_train.append(ARCExample(new_input, new_output))
        
        new_test = []
        for ex in task.test_examples:
            new_input = self._scale_grid(ex.input_grid)
            new_output = self._scale_grid(ex.output_grid) if ex.output_grid is not None else None
            new_test.append(ARCExample(new_input, new_output))
        
        return ARCTask(task.name + f"_scale{self.scale_factor}", new_train, new_test)
    
    def _scale_grid(self, grid: np.ndarray) -> np.ndarray:
        """Scale up a grid by repeating each cell."""
        if grid is None:
            return None
        return np.repeat(np.repeat(grid, self.scale_factor, axis=0), self.scale_factor, axis=1)


class FlipAugmenter(ARCAugmenter):
    """Flip grids along specified axis."""
    
    def __init__(self, axis: int):
        self.axis = axis
        self.axis_name = "horizontal" if axis == 0 else "vertical"
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        new_train = []
        for ex in task.train_examples:
            new_input = np.flip(ex.input_grid, axis=self.axis)
            new_output = np.flip(ex.output_grid, axis=self.axis) if ex.output_grid is not None else None
            new_train.append(ARCExample(new_input, new_output))
        
        new_test = []
        for ex in task.test_examples:
            new_input = np.flip(ex.input_grid, axis=self.axis)
            new_output = np.flip(ex.output_grid, axis=self.axis) if ex.output_grid is not None else None
            new_test.append(ARCExample(new_input, new_output))
        
        return ARCTask(task.name + f"_flip_{self.axis_name}", new_train, new_test)


class PermuteColorsAugmenter(ARCAugmenter):
    """Randomly permute colors in grids."""
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        # Find all unique colors in the task
        all_colors = set()
        for ex in task.train_examples:
            all_colors.update(ex.input_grid.flatten())
            if ex.output_grid is not None:
                all_colors.update(ex.output_grid.flatten())
        
        for ex in task.test_examples:
            all_colors.update(ex.input_grid.flatten())
            if ex.output_grid is not None:
                all_colors.update(ex.output_grid.flatten())
        
        colors = list(all_colors)
        if len(colors) <= 1:
            return task  # No permutation needed
        
        # Create random permutation
        permuted_colors = colors.copy()
        rng.shuffle(permuted_colors)
        color_map = {old: new for old, new in zip(colors, permuted_colors)}
        
        def apply_color_map(grid):
            if grid is None:
                return None
            new_grid = grid.copy()
            for old_color, new_color in color_map.items():
                new_grid[grid == old_color] = new_color
            return new_grid
        
        new_train = []
        for ex in task.train_examples:
            new_input = apply_color_map(ex.input_grid)
            new_output = apply_color_map(ex.output_grid)
            new_train.append(ARCExample(new_input, new_output))
        
        new_test = []
        for ex in task.test_examples:
            new_input = apply_color_map(ex.input_grid)
            new_output = apply_color_map(ex.output_grid)
            new_test.append(ARCExample(new_input, new_output))
        
        return ARCTask(task.name + "_permuted", new_train, new_test)


class CropAugmenter(ARCAugmenter):
    """Crop grids to smaller size while preserving the pattern."""
    
    def __init__(self, crop_ratio: float = 0.8):
        self.crop_ratio = crop_ratio
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        new_train = []
        for ex in task.train_examples:
            new_input = self._crop_grid(ex.input_grid, rng)
            new_output = self._crop_grid(ex.output_grid, rng) if ex.output_grid is not None else None
            if new_input is not None and (new_output is not None or ex.output_grid is None):
                new_train.append(ARCExample(new_input, new_output))
        
        new_test = []
        for ex in task.test_examples:
            new_input = self._crop_grid(ex.input_grid, rng)
            new_output = self._crop_grid(ex.output_grid, rng) if ex.output_grid is not None else None
            if new_input is not None and (new_output is not None or ex.output_grid is None):
                new_test.append(ARCExample(new_input, new_output))
        
        if not new_train:
            return task  # Fallback if cropping failed
        
        return ARCTask(task.name + "_cropped", new_train, new_test)
    
    def _crop_grid(self, grid: np.ndarray, rng: np.random.RandomState) -> Optional[np.ndarray]:
        """Crop a grid while preserving important content."""
        if grid is None:
            return None
        
        h, w = grid.shape
        new_h = max(1, int(h * self.crop_ratio))
        new_w = max(1, int(w * self.crop_ratio))
        
        if new_h >= h and new_w >= w:
            return grid
        
        # Try to find a good crop position
        start_h = rng.randint(0, h - new_h + 1)
        start_w = rng.randint(0, w - new_w + 1)
        
        return grid[start_h:start_h + new_h, start_w:start_w + new_w]


class PadAugmenter(ARCAugmenter):
    """Pad grids with background color."""
    
    def __init__(self, pad_size: int = 2):
        self.pad_size = pad_size
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        new_train = []
        for ex in task.train_examples:
            new_input = self._pad_grid(ex.input_grid, rng)
            new_output = self._pad_grid(ex.output_grid, rng) if ex.output_grid is not None else None
            new_train.append(ARCExample(new_input, new_output))
        
        new_test = []
        for ex in task.test_examples:
            new_input = self._pad_grid(ex.input_grid, rng)
            new_output = self._pad_grid(ex.output_grid, rng) if ex.output_grid is not None else None
            new_test.append(ARCExample(new_input, new_output))
        
        return ARCTask(task.name + "_padded", new_train, new_test)
    
    def _pad_grid(self, grid: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Pad a grid with the most common color (background)."""
        if grid is None:
            return None
        
        # Find most common color (background)
        unique_colors, counts = np.unique(grid, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]
        
        # Add padding
        pad_width = ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size))
        return np.pad(grid, pad_width, mode='constant', constant_values=bg_color)


class ChainAugmenter(ARCAugmenter):
    """Apply a sequence of augmentations."""
    
    def __init__(self, augmenters: List[ARCAugmenter], max_chain_length: int = 3):
        self.augmenters = augmenters
        self.max_chain_length = max_chain_length
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        # Choose random chain length
        chain_length = rng.randint(1, min(self.max_chain_length, len(self.augmenters)) + 1)
        
        # Choose random sequence of augmenters
        chain = rng.choice(self.augmenters, size=chain_length, replace=False)
        
        # Apply chain
        result_task = task
        for augmenter in chain:
            try:
                result_task = augmenter.apply_to_task(result_task, rng)
                
                # Check size constraints after each step
                if not self.check_size_constraints(result_task):
                    return task  # Return original if size exceeded
                    
            except Exception as e:
                logging.debug(f"Chain augmentation failed: {e}")
                return task  # Return original if any step fails
        
        return result_task


class RepeatAugmenter(ARCAugmenter):
    """Apply the same augmentation multiple times."""
    
    def __init__(self, base_augmenter: ARCAugmenter, max_repeats: int = 3):
        self.base_augmenter = base_augmenter
        self.max_repeats = max_repeats
    
    def apply_to_task(self, task: ARCTask, rng: np.random.RandomState) -> ARCTask:
        # Choose random number of repeats
        repeats = rng.randint(1, self.max_repeats + 1)
        
        result_task = task
        for i in range(repeats):
            try:
                result_task = self.base_augmenter.apply_to_task(result_task, rng)
                
                # Check size constraints after each repeat
                if not self.check_size_constraints(result_task):
                    return task  # Return original if size exceeded
                    
            except Exception as e:
                logging.debug(f"Repeat augmentation failed: {e}")
                return task  # Return original if any repeat fails
        
        return result_task


class ARCAugmentationManager:
    """
    Manages the full augmentation pipeline for ARC tasks.
    
    This implements the configuration-based augmentation system from SEAL.
    """
    
    def __init__(self):
        # Basic augmenters (matching original SEAL exactly)
        self.basic_augmenters = [
            RotateAugmenter(90),
            RotateAugmenter(180),
            RotateAugmenter(270),
            FlipAugmenter(0),  # Horizontal flip
            FlipAugmenter(1),  # Vertical flip
            TransposeAugmenter(),
            PermuteColorsAugmenter(),
        ]
        
        # Size augmenters (matching original SEAL)
        self.size_augmenters = [
            IncreaseResolutionAugmenter(2),
            CropAugmenter(0.8),
            CropAugmenter(0.9),
            PadAugmenter(1),
            PadAugmenter(2),
        ]
        
        # Chain augmenters (combinations of basic + size)
        self.chain_augmenters = [
            ChainAugmenter([RotateAugmenter(90), IncreaseResolutionAugmenter(2)], max_chain_length=2),
            ChainAugmenter([RotateAugmenter(270), IncreaseResolutionAugmenter(2)], max_chain_length=2),
            ChainAugmenter([RotateAugmenter(180), IncreaseResolutionAugmenter(2)], max_chain_length=2),
            ChainAugmenter([FlipAugmenter(0), IncreaseResolutionAugmenter(2)], max_chain_length=2),
            ChainAugmenter([FlipAugmenter(1), IncreaseResolutionAugmenter(2)], max_chain_length=2),
            ChainAugmenter([TransposeAugmenter(), IncreaseResolutionAugmenter(2)], max_chain_length=2),
        ]
    
    def get_augmenters_from_config(self, config: Dict[str, Any]) -> List[ARCAugmenter]:
        """Get augmenters based on configuration (matching original SEAL exactly)."""
        augmenters = []
        data_gen = config.get("data_generation", {})
        
        # Basic augmentations (matching original SEAL)
        if data_gen.get("use_basic_augmentations", False):
            augmenters.extend(self.basic_augmenters)
        
        # Size augmentations (matching original SEAL)  
        if data_gen.get("use_size_augmentations", False):
            augmenters.extend(self.size_augmenters)
        
        # Chain augmentations (matching original SEAL)
        if data_gen.get("use_chain_augmentations", False):
            augmenters.extend(self.chain_augmenters)
        
        # Repeat augmentations (matching original SEAL)
        if data_gen.get("use_repeat_augmentations", False):
            for base_aug in self.basic_augmenters[:4]:  # Limit to core augmentations
                repeat_augmenter = RepeatAugmenter(base_aug, max_repeats=2)
                augmenters.append(repeat_augmenter)
        
        return augmenters
    
    def check_size_constraints(self, task: ARCTask, max_size: int = 30) -> bool:
        """Check that task meets size constraints."""
        for ex in task.train_examples + task.test_examples:
            if ex.input_grid.shape[0] > max_size or ex.input_grid.shape[1] > max_size:
                return False
            if ex.output_grid is not None and (ex.output_grid.shape[0] > max_size or ex.output_grid.shape[1] > max_size):
                return False
        return True
    
    def apply_config_to_task(self, task: ARCTask, config: Dict[str, Any], max_tasks: int = 250) -> List[ARCTask]:
        """
        Apply configuration to generate augmented training data.
        
        This follows the original SEAL methodology of creating leave-one-out tasks
        and applying augmentations to them.
        """
        augmenters = self.get_augmenters_from_config(config)
        
        # Generate leave-one-out tasks like original SEAL
        train_tasks = []
        
        # For each training example, create tasks with that example held out
        for i in range(len(task.train_examples)):
            # Create task with all examples except i
            remaining_examples = [ex for j, ex in enumerate(task.train_examples) if j != i]
            held_out_example = task.train_examples[i]
            
            # Create new task with held-out example as test
            leave_one_out_task = ARCTask(
                name=f"{task.name}_leave_{i}",
                train_examples=remaining_examples,
                test_examples=[ARCExample(held_out_example.input_grid, held_out_example.output_grid)]
            )
            
            # Add original task
            train_tasks.append(leave_one_out_task)
            
            # Apply augmentations
            for j, augmenter in enumerate(augmenters):
                try:
                    rng = np.random.RandomState(42 + i * 100 + j)  # Deterministic for reproducibility
                    augmented_task = augmenter.apply_to_task(leave_one_out_task, rng)
                    
                    # Check size constraints
                    if self.check_size_constraints(augmented_task):
                        train_tasks.append(augmented_task)
                    
                    # Limit number of tasks to prevent memory issues
                    if len(train_tasks) >= max_tasks:
                        break
                        
                except Exception as e:
                    logging.debug(f"Augmentation failed: {e}")
                    continue
            
            if len(train_tasks) >= max_tasks:
                break
        
        # Remove duplicates based on task name
        unique_tasks = list({task.name: task for task in train_tasks}.values())
        
        logging.info(f"Generated {len(unique_tasks)} augmented tasks from config")
        return unique_tasks[:max_tasks]


def test_augmentation_manager():
    """Test the augmentation manager."""
    logging.basicConfig(level=logging.INFO)
    
    # Create mock ARC task
    mock_input = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mock_output = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    
    task = ARCTask(
        name="test_task",
        train_examples=[
            ARCExample(mock_input, mock_output),
            ARCExample(mock_input * 2, mock_output * 2)
        ],
        test_examples=[ARCExample(mock_input, None)]
    )
    
    # Test configuration
    config = {
        "data_generation": {
            "use_basic_augmentations": True,
            "use_size_augmentations": True,
            "use_chain_augmentations": False,
            "use_repeat_augmentations": False
        }
    }
    
    # Apply augmentations
    manager = ARCAugmentationManager()
    augmented_tasks = manager.apply_config_to_task(task, config)
    
    print(f"Generated {len(augmented_tasks)} augmented tasks")
    for aug_task in augmented_tasks[:5]:  # Show first 5
        print(f"  - {aug_task.name}")


if __name__ == "__main__":
    test_augmentation_manager()
