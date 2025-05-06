import torch
from torch import nn
from torch.nn import Module
from grail_metabolism.utils.preparation import MolFrame
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score as roc_auc, matthews_corrcoef as mcc
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple


def cpunum(tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


@torch.no_grad()
def test(model: Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    pred = []
    bin_pred = []
    real = []
    for data in tqdm(loader):
        out = model(data)
        pred.extend(list(cpunum(out)))
        bin_pred.extend(list((cpunum(out) > 0.5).astype(int)))
        real.extend(list(cpunum(data.y)))
    mat = mcc(real, bin_pred)
    roc = roc_auc(real, pred)
    return mat, roc


def train_pairs(model: Module, train_set: MolFrame, test_set: MolFrame, lr: float, decay: float, eps: int) -> Module:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

    train_loader = []
    for pairs in train_set.graphs.values():
        for pair in pairs:
            if pair is not None:
                train_loader.append(pair)
    train_loader = DataLoader(train_loader, batch_size=128, shuffle=True)

    test_loader = []
    for pairs in test_set.graphs.values():
        for pair in pairs:
            if pair is not None:
                test_loader.append(pair)
    test_loader = DataLoader(test_loader, batch_size=128)

    history = []
    for epoch in tqdm(range(eps)):
        model.train()
        for batch in train_loader:
            out = model(batch)
            loss = criterion(out, batch.y.unsqueeze(1).float())
            history.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            mat, roc = test(test_loader)
            plt.plot(history)
            print(f'Epoch {epoch + 1}: MCC = {mat:.4f}, ROC = {roc:.4f}')

    return model


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=True):
        super(PULoss, self).__init__()

        if not 0 < prior < 1:
            raise TypeError("The class prior should be in (0, 1)")

        self.prior = torch.tensor(prior)
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = 0
        self.min_count = 1

    def forward(self, inp, target):
        assert (inp.shape == target.shape)

        if inp.is_cuda:
            self.prior = self.prior.cuda()

        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        n_positive, n_unlabeled = torch.clamp(torch.sum(positive), min=self.min_count), torch.clamp(
            torch.sum(unlabeled), min=self.min_count)

        y_positive = self.loss_func(inp) * positive
        y_positive_inv = self.loss_func(-inp) * positive
        y_unlabeled = self.loss_func(-inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk

        return positive_risk + negative_risk