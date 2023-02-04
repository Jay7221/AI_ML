import torch
import numpy as np
import copy
# x = torch.empty(10)
# print(x)

# x = torch.ones(2, 2, dtype = torch.float16)

# print(x)
# print(x.dtype)
# print(x.size())


# x = torch.rand([3, 2])
# y = torch.rand((3, 2))

# z = x + y
# x += y
# print(x)
# print(y)
# print(z)


# z = x * y
# print(z)
# z = torch.multiply(x, y)
# print(z)


# print(x[1][1].item())

# x = torch.rand(4, 4)``
# print(x)

# y = x.view(-1, 8)
# print(y)


# a = torch.ones(5)
# b = copy.deepcopy( a.numpy())
# a.add_(1)
# print(a)
# print(b)


# a = np.ones(5)
# b = torch.from_numpy(a)
# print(a)
# print(b)

import time

if torch.cuda.is_available():
    device = torch.device("cuda")
    sz =  10000000
    x = torch.rand(sz)
    y = torch.rand(sz)
    start = time.time()
    print("***************************** CPU Speed *****************************")
    z = x * y
    print(time.time() - start, "with device", z.device)
    
    
    
    x_gpu = x.to(device)
    y_gpu = y.to(device)
    torch.cuda.synchronize()
    for i in range(3):
        start = time.time()
        print("***************************** GPU Speed *****************************")
        z_gpu = torch.mul(x_gpu, y_gpu)
        print(time.time() - start, "with device", z_gpu.device)