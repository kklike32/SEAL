# Simple test to check if basic MLX gradient computation works
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam

def simple_test():
    """Test basic MLX gradient computation"""
    
    # Create a simple linear model
    model = nn.Linear(10, 1)
    
    # Create some dummy data
    x = mx.random.normal((5, 10))
    y = mx.random.normal((5, 1))
    
    # Define loss function
    def loss_fn():
        pred = model(x)
        return mx.mean((pred - y) ** 2)
    
    # Compute gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn()
    
    print(f"Loss: {loss}")
    print(f"Gradients computed: {len(grads)} parameters")
    print("Basic MLX gradient computation works!")

if __name__ == '__main__':
    simple_test()
