# Pytorch Tensor Tutorial
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor=torch.tensor([[1, 2, 3],[4, 5, 6]], dtype=torch.float32,
                       device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)

#common initialization methods
# 1. Creating an empty stensor 
x = torch.empty(size=(3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(3, 3)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=5, steps=10)
x = torch.diag(torch.ones(3))

# 2. Uniform distribution
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x = torch.empty(size=(1, 5)).uniform_(0,1)
print(x)

# 3. How to initialize and convert to diffrernt types
tensor = torch.arange(4)
print(tensor)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

## 4. Array to tensor conversion
import numpy as np
np_array =np.zeros((5,5))
tensor = torch.from_numpy(np_array)
print(tensor)
np_array_back = tensor.numpy()
print(np_array_back)

# 5. Tensor Math and Comparison Operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# 5.1 Addition
z1 = torch.empty(3)
# Method 1
torch.add(x, y, out=z1)
print(z1)
#Method 2
z2 = torch.add(x, y)
print(z2)
#Method 3
z = x + y
print(z)

# 5.2 Subtraction
z3 = y - x
print(z3)

# 5.3 Division
z4= torch.true_divide(x, y) # Elementwise element if x and y of equal shape
print(z4)

# 5.4 inplace operations
t = torch.zeros(3)
t.add_(x)
t += x
print(t)

# 5.5 Exponentiation
z5 = x.pow(2)
z6 = x ** 2
print(z6)

# 5.6 Comparisons
z5 = x > 0
z6 = x < 0
print(z5)

# 5.7 Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # the following expression is also equivlent to this one
x4 = x1.mm(x2)
print(x3)
print(x4)

# 5.8 Matrix exponentiation
matrix_exp = torch.rand(4,4)
print(matrix_exp.matrix_power(3))

# 5.9 Element wise multi
z = x * y
print(x)
print(y)
print(z)

# 5.10 Dot product
z = torch.dot(x, y)
print(z)

# 5.11 Batch Matrix Multiplication
batch = 3
n= 4
m = 5
p = 2
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(tensor1)
print(tensor2)
print(out_bmm)

# 6. Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z1 = x1 - x2 # in this case x2 will be broadcasted to shape (5,5)
z2 = x1 ** x2
print(z1)
print(z2)

# 7. Other useful tensor operations
x1 = torch.rand((1,4))
sum_x=torch.sum(x1, dim=1)

values, indices = torch.max(x1, dim=1)
values, indices = torch.min(x1, dim=1)
abs_x = torch.abs(x1)
print(abs_x)

mean_x = torch.mean(x1.float(), dim=0)
print(mean_x)

# 8. Elementwise comparison
z = torch.eq(x, y)
print(z)

# 9. Sorting tensor elements
sort_y, indices = torch.sort(y, dim=0, descending=False)
print(sort_y, indices)

# 10. Extra

z = torch.clamp(x, min=0)
x= torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x) # this means any of x value need to be true
z = torch.all(x) # this means all of x values need to be true
print(z)

# 11. Tensor Indexing
batch_size = 10
features = 25
x = torch.rand((batch_size,features))
print(x[0].shape)
print(x[:, 0].shape)

# 11.1 To get the third example in the batch and the first 10 features
#print(x)
print(x[2, 0:10])
x[0, 0]= 0
print(x)

# 11.2 Fancy indexing
x = torch.arange(10)
indices=[1, 4, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])

#11.3 More advanced Indexing
x = torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[x.remainder(2) == 0])

# 11.4. Usefull operation
print(torch.where(x>5, x, x*2)) # this means: (if x < 5 print x*2 else print x)
print(torch.tensor([0,1,1,2,2,3,4,5]).unique()) # to return the unique values

x = torch.rand((3, 5))
print(x.ndimension()) # tensor dimensions
print(x.numel())      # number of elements in a given tensor x

# 12. Reshaping tensors
x = torch.arange(9)
x1 = x.reshape(3, 3)
x1 = x.view(3, 3)
print(x1.shape)

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0))
print(x1)
z = x1.view(-1) # to flat x1
print(z)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)
print(x.shape)

z = x.permute(0, 2, 1) # to exchange the dimensions
print(z.shape)

x = torch.arange(10)
print(x)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)