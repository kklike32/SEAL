# knowledge-incorporation/src/utils_mlx.py
import logging
from typing import Any, Dict, List, Tuple
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import generate
from mlx_lm.tuner import linear_to_lora_layers 
from mlx_lm.sample_utils import make_sampler

# Import reusable components from the original utils file
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

# def accuracy_and_texts_mlx(
#     model: Any,
#     tokenizer: Any,
#     questions: List[Dict[str, str]],
#     sampling_config: Dict[str, Any],
#     instruct_model: bool
# ) -> Tuple[float, List[str], List[bool]]:
#     LOG.error("Using base 0's, not GPT")
#     return 0, 0, 0

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

# ============================================================================== #
# SECTION 2: TRAINING UTILITIES (NEW)
# ============================================================================== #

def extend_and_pad(sequence: List[int], max_length: int, pad_value: int) -> List[int]:
    """Your excellent custom padding and truncation function."""
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        padding = [pad_value] * (max_length - len(sequence))
        return sequence + padding

# def loss_fn(model, inputs, targets):
#     """Standard cross-entropy loss for MLX, respecting our -100 mask."""
#     logits = model(inputs)
#     logits = logits.astype(mx.float32)
#     return nn.losses.cross_entropy(logits, targets, reduction="mean")

IGNORE_INDEX = -100

def masked_ce_loss(model, inputs, targets):
    """
    A custom cross-entropy loss function for MLX that correctly handles an ignore_index.
    This prevents the `IndexError: vector` crash caused by negative labels.
    """
    # Handle models that might return logits as a tuple (e.g., with past_key_values)
    output = model(inputs)
    logits = output[0] if isinstance(output, tuple) else output
    logits = logits.astype(mx.float32)

    # Create a mask to identify tokens that should be ignored
    mask = (targets != IGNORE_INDEX)
    
    # If all tokens are ignored in the batch, return zero loss
    if not mx.any(mask):
        return mx.array(0.0, dtype=mx.float32)

    # To prevent indexing errors, replace ignored labels (-100) with a valid index (0).
    # The mask ensures these positions do not contribute to the final loss.
    safe_targets = mx.where(mask, targets, mx.zeros_like(targets))
    
    # Calculate the cross-entropy for each token individually
    per_token_loss = nn.losses.cross_entropy(logits, safe_targets, reduction="none")
    
    # Apply the mask to the per-token losses and calculate the mean loss
    # over the non-ignored tokens.
    masked_loss = per_token_loss * mask.astype(per_token_loss.dtype)
    final_loss = masked_loss.sum() / mask.sum()
    
    return final_loss

def run_lora_training(
    model: nn.Module,
    tokenizer: Any,
    train_sequences: List[str],
    finetune_args: Dict[str, Any],
    lora_config: Dict[str, Any],
    max_seq_length: int
) -> nn.Module:
    """
    Performs a full LoRA fine-tuning run using a manual training loop.
    This gives us full control over the data and loss calculation.
    """
    # 1. Freeze model and apply LoRA layers
    model.freeze()
    linear_to_lora_layers(model, finetune_args['lora_layers'], lora_config)

    # 2. Prepare and pad the dataset, respecting the mask
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    sub_ids = tokenizer.encode(finetune_args['end_mask_substring'])
    
    inputs, labels = [], []
    for seq in train_sequences:
        tokens = tokenizer.encode(seq)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        
        # Find mask position and apply mask
        mask_pos = -1
        M = len(sub_ids)
        for i in range(len(tokens) - M + 1):
            if tokens[i : i + M] == sub_ids:
                mask_pos = i + M
                break
        
        current_labels = list(tokens)
        if mask_pos != -1:
            current_labels[:mask_pos] = [-100] * mask_pos
        
        # Pad both to the max length
        inputs.append(extend_and_pad(tokens, max_seq_length, pad_id))
        labels.append(extend_and_pad(current_labels, max_seq_length, -100))

    if not inputs:
        LOG.warning("No training data was processed. Skipping training.")
        model.unfreeze()
        return model

    # Convert to MLX arrays for training
    inputs = mx.array(inputs)
    labels = mx.array(labels)
    
    # 3. Set up the training components
    optimizer = optim.Adam(learning_rate=finetune_args['finetune_lr'])
    
    # Create the gradient function using new custom masked loss function.
    loss_and_grad_fn = nn.value_and_grad(model, masked_ce_loss)
    
    # 4. The Training Loop
    LOG.info(f"Starting manual training loop for {finetune_args['finetune_epochs']} epochs...")
    batch_size = finetune_args['batch_size']
    model.train()
    for epoch in range(finetune_args['finetune_epochs']):
        epoch_loss = 0.0
        num_batches = 0
        # Create batches for the current epoch
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            # LOG.info(f"Running batch {num_batches}... Shape of inputs: {batch_inputs.shape}, Dtype: {batch_inputs.dtype}")
            # LOG.info(f"Shape of labels: {batch_labels.shape}, Dtype: {batch_labels.dtype}")
            
            # Get loss and gradients, then update model
            loss, grads = loss_and_grad_fn(model, batch_inputs, batch_labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            LOG.info(f"Epoch {epoch + 1}/{finetune_args['finetune_epochs']} | Average Loss: {avg_loss:.4f}")

    model.unfreeze()
    return model

