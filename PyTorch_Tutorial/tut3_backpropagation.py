import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# Forward pass and compute the loss
y_pred = x * w
loss = (y_pred - y) ** 2


# Backward pass
loss.backward()

# Update weights
lr = 0.2
for epoch in range(300):
    w.grad.zero_()
    y_pred = w * x
    loss = (y_pred - y) ** 2
    loss.backward()
    with torch.no_grad():
        print(w.grad)
        k = lr * w.grad
        w -= k

print("Done!")
print(w * x)