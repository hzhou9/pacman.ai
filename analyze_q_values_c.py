import torch
import torch.nn as nn
import numpy as np
import argparse
from DQN_color import DQN  # Import from your standalone file

def generate_dummy_states(batch_size=32, height=168, width=168, channels=12):
    """Generate random RGB frames for testing (12 channels: 4 frames × 3 RGB)."""
    return torch.rand(batch_size, channels, height, width)

def analyze_q_values(model_path):
    """Load a saved DQN model and analyze Q-value convergence metrics."""
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize DQN with RGB parameters (12 channels)
    model = DQN(input_shape=(12, 168, 168), num_actions=4).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create a target network
    target_net = DQN(input_shape=(12, 168, 168), num_actions=4).to(device)
    target_net.load_state_dict(state_dict)
    target_net.eval()
    
    # Generate dummy states for analysis
    batch_size = 32
    states = generate_dummy_states(batch_size).to(device)  # [32, 12, 168, 168]
    
    # Compute Q-values from policy and target networks
    with torch.no_grad():
        policy_q_values = model(states)  # [batch_size, num_actions]
        target_q_values = target_net(states)  # Same for comparison
    
    # Calculate Q-value distribution
    q_mean = policy_q_values.mean().item()
    q_std = policy_q_values.std().item()
    q_max = policy_q_values.max().item()
    
    # Difference between policy and target Q-values
    q_diff_mean = torch.abs(policy_q_values - target_q_values).mean().item()
    
    # Simulate action consistency (assuming epsilon = 0 for inference)
    action_consistency = (policy_q_values.argmax(dim=1) == target_q_values.argmax(dim=1)).float().mean().item()
    
    # Hypothetical loss (MSE between policy and target for these states)
    hypothetical_loss = nn.MSELoss()(policy_q_values, target_q_values).item()
    
    # Print results
    print(f"Model Analysis for {model_path}:")
    print(f"Q-Value Mean: {q_mean:.4f}")
    print(f"Q-Value Std: {q_std:.4f}")
    print(f"Q-Value Max: {q_max:.4f}")
    print(f"Policy-Target Q-Diff Mean: {q_diff_mean:.4f}")
    print(f"Action Consistency: {action_consistency:.3f}")
    print(f"Hypothetical Loss: {hypothetical_loss:.4f}")
    
    # Interpretation
    if q_std < 0.5 and q_diff_mean < 1.0 and hypothetical_loss < 0.1:
        print("Q-values appear to be converging (stable and aligned).")
    else:
        print("Q-values may not be converging (high variance, difference, or loss).")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Q-value convergence of a saved DQN model.")
    parser.add_argument("-m", "--model-path", type=str, required=True,
                        help="Path to the saved .pth model file (e.g., pacman_dqn_3000.pth)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    analyze_q_values(args.model_path)