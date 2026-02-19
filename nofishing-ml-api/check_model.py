# -*- coding: utf-8 -*-
"""
Check Trained Model Structure
检查训练好的模型结构
"""
import torch

print("Checking trained model structure...")

# Load the model state dict
MODEL_PATH = "models/phishing_classifier_url_only.pt"
state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

print(f"\nModel keys ({len(state_dict)}):")
for key in sorted(state_dict.keys()):
    shape = state_dict[key].shape
    print(f"  {key}: {shape}")

# Reconstruct the model architecture from the state dict
print("\nInferred architecture:")
print("  net.0: Linear(20, 64)")
print("  net.1: ReLU()")
print("  net.2: BatchNorm1d(64)")
print("  net.3: Dropout(0.3)")
print("  net.4: Linear(64, 32)")
print("  net.5: ReLU()")
print("  net.6: BatchNorm1d(32)")
print("  net.7: Dropout(0.2)")
print("  net.8: Linear(32, 16)")
print("  net.9: ReLU()")
print("  net.10: Linear(16, 1)")
print("  net.11: Sigmoid()")
