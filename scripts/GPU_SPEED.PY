import torch

x = torch.rand(10000, 10000, device="cuda")
y = torch.rand(10000, 10000, device="cuda")

z = x @ y

print(z.shape)
print(torch.cuda.get_device_name(0))