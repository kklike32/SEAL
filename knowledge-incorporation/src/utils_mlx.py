# knowledge-incorporation/src/utils_mlx.py
import logging
from typing import Any, Dict, List, Tuple

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

# Import reusable components from the original utils file
# This avoids code duplication and keeps our new file focused.
from .utils import (
    format_answer_prompts,
    format_grade_prompts,
    grade_with_gpt4
)

LOG = logging.getLogger(__name__)

def generate_mlx(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    sampling_config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    An MLX-native replacement for the VLLM-based generate function.

    Args:
        model: The loaded MLX model.
        tokenizer: The loaded tokenizer.
        prompts: A list of prompts to generate completions for.
        sampling_config: A dictionary with keys like 'temperature', 'top_p', 'max_tokens'.

    Returns:
        A list of dictionaries, mirroring the original VLLM output format,
        e.g., [{'text': 'completion_one'}, {'text': 'completion_two'}].
    """
    outputs = []
    
    # Create a sampler object from the configuration. This is the correct
    # way to pass temperature/top_p to the mlx_lm generator.
    sampler = make_sampler(
        temp=sampling_config.get("temperature", 0.0),
        top_p=sampling_config.get("top_p", 1.0)
    )

    LOG.info(f"Generating {len(prompts)} completions with MLX...")
    for prompt in prompts:
        # Note: mlx_lm.generate doesn't support batching, so we loop.
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=sampling_config.get("max_tokens", 64),
            sampler=sampler,
        )
        outputs.append({"text": response})
    LOG.info("Generation complete.")
    return outputs


def accuracy_and_texts_mlx(
    model: Any,
    tokenizer: Any,
    questions: List[Dict[str, str]],
    sampling_config: Dict[str, Any],
    instruct_model: bool
) -> Tuple[float, List[str], List[bool]]:
    """
    MLX-native version of the accuracy evaluation function.

    This function orchestrates the process of generating answers with MLX
    and then grading them using the imported GPT-4 grading logic.
    """
    if not questions:
        return 0.0, [], []

    # 1. Generate answers using the provided MLX model
    prompts = format_answer_prompts(questions, instruct_model=instruct_model)
    ans_out = generate_mlx(model, tokenizer, prompts, sampling_config)
    
    # Ensure preds is a list of strings, handling potential empty responses
    preds = [o.get("text", "").strip() for o in ans_out]
    LOG.debug(f"Generated predictions: {preds}")

    # 2. Grade the answers using the imported GPT-4 grading utility
    verdicts: List[bool] = [False] * len(preds)
    
    # Filter out empty predictions to avoid sending them to the expensive GPT-4 API
    q_sub, p_sub, idx_sub = [], [], []
    for i, (q, p) in enumerate(zip(questions, preds)):
        if p:  # Only grade if the prediction is not empty
            q_sub.append(q)
            p_sub.append(p)
            idx_sub.append(i)

    if q_sub:
        LOG.info(f"Grading {len(q_sub)} non-empty predictions with GPT-4...")
        grade_prompts = format_grade_prompts(q_sub, p_sub)
        graded_verdicts = grade_with_gpt4(grade_prompts)
        for i, v in zip(idx_sub, graded_verdicts):
            verdicts[i] = v
        LOG.info("Grading complete.")
    
    # 3. Calculate final accuracy
    acc = sum(verdicts) / len(questions) if questions else 0.0
    return acc, preds, verdicts
