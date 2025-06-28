import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mx_utils
import sys

# Import the components we need to test from your utils_mlx file
from utils_mlx import masked_ce_loss, IGNORE_INDEX

def run_test():
    """
    Runs a minimal unit test on the masked_ce_loss function.
    """
    print("--- Running Sanity Check for masked_ce_loss ---")

    # 1. Define a minimal dummy model. An nn.Linear layer is sufficient.
    # It just needs to be an nn.Module that can be called.
    # vocab_size = 10, embedding_dim = 4
    dummy_vocab_size = 10
    model = nn.Embedding(dummy_vocab_size, 4)
    model.train() # Set to training mode

    # 2. Define the test inputs and labels as requested
    ids  = mx.array([[1, 2, 3]])          # (batch_size=1, seq_len=3)
    labs = mx.array([[1, 2, IGNORE_INDEX]]) # Labels with an ignore token

    print(f"Test Input IDs:  {ids}")
    print(f"Test Labels:     {labs}")
    print(f"Ignore Index:    {IGNORE_INDEX}")

    # 3. Create the value and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, masked_ce_loss)

    # 4. Perform a single forward-backward step
    try:
        # Use the correct signature, passing the model as the first argument
        loss, grads = loss_and_grad_fn(model, ids, labs)
        mx.eval(loss, grads) # Force evaluation

        print("\n--- ✅ SUCCESS ---")
        print(f"Calculated Loss: {loss.item():.4f}")
        
        # Check that gradients were computed
        grad_tree = mx_utils.tree_flatten(grads)
        if grad_tree:
            print(f"Gradients were computed for {len(grad_tree)} parameter groups.")
            # print("Gradient sample:", grad_tree[0][1])
        else:
            print("Warning: No gradients were computed.")

    except Exception as e:
        print("\n--- ❌ FAILED ---")
        print(f"An error occurred during the sanity check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()

