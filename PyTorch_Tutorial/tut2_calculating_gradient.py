import torch

x = torch.rand(3, requires_grad=True)

# print(x)
# y = x + 2
# z = y * y * 2
# z = z.mean()
# print(z)

# z.backward()
# print(x.grad)

# x.requries_grad_(False)
# x.detach()
# with torch.no_grad():
#   oprations


# x.requires_grad_(False)
# y = x.detach()
# print(y)
# print(x)

# with torch.no_grad():
#     y = x + 2
#     z = (y * y * x + 43) * 4
#     print(y)
#     print(z)
#     print(x)


weights = torch.ones([4], requires_grad=True)

for epoch in range(4):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()