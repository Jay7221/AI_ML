import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("softmax numpy: ", outputs)





x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print("softmax torch: ", outputs)



def cross_entropy(actual, predicted):
    loss = - np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1, 0, 0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')


loss = nn.CrossEntropyLoss()

Y = torch.tensor([2, 0, 1])
# number_samples * number_classes = 1 X 3
Y_pred_good = torch.tensor([[2.0, 1.0, 10], [2.0, 1.0, 0.1], [2.0, 100, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3], [2.0, 1.0, 0.1], [2.0, 1.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'Loss1 torch: {l1.item():.4f}')
print(f'Loss2 torch: {l2.item():.4f}')

_, preditctions1 = torch.max(Y_pred_good, 1)
_, preditctions2 = torch.max(Y_pred_bad, 1)

print(preditctions1)
print(preditctions2)