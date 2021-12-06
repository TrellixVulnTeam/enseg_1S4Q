import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import TensorDataset, dataloader
from torch.utils.data import DataLoader

w = Variable(torch.tensor([0.5, 0.5, 0.5]).cuda(), requires_grad=True)
b = Variable(torch.tensor([0.0, 0.0, 0.0]).cuda(), requires_grad=True)
X = torch.linspace(0, 10, 1000).cuda()
Y = 2 * X ** 3 + 10 * X ** 2 + 100 * X.cuda()
creterion = nn.MSELoss()
opt = torch.optim.Adam([w, b], 1e-1)
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, 128, True)
for e in range(50):
    sum_loss = 0
    for i, (x, y) in enumerate(dataloader):
        pred = (
            b[0] * torch.pow(x, w[0])
            + b[1] * torch.pow(x, w[1])
            + b[2] * torch.pow(x, w[2])
        )
        loss = creterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sum_loss += loss.item()
    print(f"epoch {e},loss: {sum_loss/len(X)}")
import matplotlib.pyplot as plt

w, b, X, Y = w.cpu(), b.cpu(), X.cpu(), Y.cpu()
plt.figure(1)
plt.plot(X.numpy(), Y.numpy())
plt.plot(
    X.numpy(),
    (b[0] * torch.pow(X, w[0]) + b[1] * torch.pow(X, w[1]) + b[2] * torch.pow(X, w[2]))
    .detach()
    .numpy(),
)
plt.legend(["ground truth", "pred"])
plt.savefig("exponentiation.png")

