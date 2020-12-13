import torch
import torch.optim as optim

# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])
#
# print(x_train)
# print(x_train.shape)
#
# w = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
# h = x_train * w + b
# print(h)
# c = torch.mean((h - y_train)**2)
# print(c)
#
# o = optim.SGD([w, b], lr=0.01)
# o.zero_grad()
# c.backward()
# o.step()

x = 2.0
w = torch.tensor(x, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    z = 2*w
    z.backward()
    print(f'수식을 w로 미분한 값 : {w.grad}')