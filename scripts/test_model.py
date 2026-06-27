import torch
from model import RouteTREEModel

# Create model
model = RouteTREEModel()

# Switch to evaluation mode
model.eval()

# Dummy input
x = torch.randn(1, 3, 512, 512)

# Disable gradient calculation
with torch.no_grad():
    output = model(x)

print("Output Shape:", output.shape)