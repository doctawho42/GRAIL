from __future__ import annotations

import time
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from ._graph import GraphEncoder
from .train_model import PULoss
from .wrapper import GFilter
from ..utils.preparation import MolFrame, build_pair_graph
from ..utils.transform import FINGERPRINT_DIM, from_rdmol


def _pad_dims(values: Sequence[int], size: int, default: int) -> List[int]:
    padded = list(values[:size])
    while len(padded) < size:
        padded.append(default)
    return padded


def _timeout_buffer(last_epoch_seconds: float) -> float:
    return max(30.0, min(300.0, 0.2 * max(last_epoch_seconds, 0.0)))


class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Data]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Data:
        return self.data[index]


class _SingleDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Tuple[Data, Data]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Data, Data]:
        return self.data[index]


class Filter(GFilter):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        arg_vec: Sequence[int],
        mode: Literal["single", "pair"],
        conv_kind: Literal["gatv2", "gcn", "gin"] = "gatv2",
        use_graph: bool = True,
        use_fingerprint: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = _pad_dims(arg_vec, 6, max(32, in_channels * 2))
        self.mode = mode
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.conv_kind = conv_kind
        self.use_graph = use_graph
        self.use_fingerprint = use_fingerprint
        self.calibrated_threshold: Optional[float] = None
        graph_dim = hidden[2] if use_graph else 0
        fp_dim = 2 * FINGERPRINT_DIM if use_fingerprint else 0
        if graph_dim + fp_dim == 0:
            raise ValueError("At least one of use_graph/use_fingerprint must be enabled")

        if mode == "pair":
            self.encoder = (
                GraphEncoder(
                    in_channels=in_channels,
                    edge_dim=edge_dim,
                    hidden_dims=hidden[:2],
                    out_dim=hidden[2],
                    conv_kind=conv_kind,
                    dropout=dropout,
                )
                if use_graph
                else None
            )
            self.classifier = nn.Sequential(
                nn.Linear(graph_dim + fp_dim, hidden[3]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden[3], hidden[4]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden[4], hidden[5]),
                nn.ReLU(inplace=True),
                nn.Linear(hidden[5], 1),
            )
        elif mode == "single":
            self.sub_encoder = (
                GraphEncoder(
                    in_channels=in_channels,
                    edge_dim=edge_dim,
                    hidden_dims=hidden[:2],
                    out_dim=hidden[2],
                    conv_kind=conv_kind,
                    dropout=dropout,
                )
                if use_graph
                else None
            )
            self.prod_encoder = (
                GraphEncoder(
                    in_channels=in_channels,
                    edge_dim=edge_dim,
                    hidden_dims=hidden[:2],
                    out_dim=hidden[2],
                    conv_kind=conv_kind,
                    dropout=dropout,
                )
                if use_graph
                else None
            )
            self.classifier = nn.Sequential(
                nn.Linear((2 * graph_dim) + fp_dim, hidden[3]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden[3], hidden[4]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden[4], 1),
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, data: Data, met: Optional[Data] = None) -> torch.Tensor:
        if self.mode == "pair":
            features = []
            device = data.x.device
            batch_size = int(data.batch.max().item()) + 1 if getattr(data, "batch", None) is not None and data.x.numel() else 1
            if self.use_graph:
                assert self.encoder is not None
                embedding = self.encoder(data)
                features.append(embedding)
                device = embedding.device
                batch_size = embedding.size(0)
            if self.use_fingerprint:
                fp = getattr(data, "fp", None)
                if fp is None:
                    fp = torch.zeros((batch_size, 2 * FINGERPRINT_DIM), device=device)
                features.append(fp.float())
            return torch.sigmoid(self.classifier(torch.cat(features, dim=1)))

        if met is None:
            raise ValueError("met is required in single mode")

        features = []
        device = data.x.device
        batch_size = int(data.batch.max().item()) + 1 if getattr(data, "batch", None) is not None and data.x.numel() else 1
        if self.use_graph:
            assert self.sub_encoder is not None and self.prod_encoder is not None
            sub_embedding = self.sub_encoder(data)
            prod_embedding = self.prod_encoder(met)
            features.extend([sub_embedding, prod_embedding])
            device = sub_embedding.device
            batch_size = sub_embedding.size(0)
        if self.use_fingerprint:
            sub_fp = getattr(data, "fp", None)
            prod_fp = getattr(met, "fp", None)
            if sub_fp is None:
                sub_fp = torch.zeros((batch_size, FINGERPRINT_DIM), device=device)
            if prod_fp is None:
                prod_fp = torch.zeros((batch_size, FINGERPRINT_DIM), device=device)
            features.extend([sub_fp.float(), prod_fp.float()])
        features = torch.cat(features, dim=1)
        return torch.sigmoid(self.classifier(features))

    def _pair_loader(self, data: MolFrame, batch_size: int, shuffle: bool = True) -> DataLoader:
        if data.graphs:
            dataset = [graph for graph in data._pair_dataset() if graph is not None]
            return DataLoader(_PairDataset(dataset), batch_size=batch_size, shuffle=shuffle)
        return data.pair_loader(batch_size=batch_size, shuffle=shuffle)

    def _single_loader(self, data: MolFrame, batch_size: int, shuffle: bool = True) -> DataLoader:
        if not data.single:
            data.singlegraphs()
        dataset = data._single_dataset()

        def collate(items: List[Tuple[Data, Data]]) -> Tuple[Batch, Batch]:
            return Batch.from_data_list([item[0] for item in items]), Batch.from_data_list([item[1] for item in items])

        return DataLoader(_SingleDataset(dataset), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    def _compute_loader_loss(self, loader: DataLoader, criterion, device: torch.device) -> float:
        total_loss = 0.0
        num_batches = 0
        self.eval()
        with torch.no_grad():
            for batch in loader:
                if self.mode == "pair":
                    batch = batch.to(device)
                    output = self(batch)
                    target = batch.y.view(-1, 1).float()
                else:
                    sub_batch, prod_batch = batch
                    sub_batch = sub_batch.to(device)
                    prod_batch = prod_batch.to(device)
                    output = self(sub_batch, prod_batch)
                    target = prod_batch.y.view(-1, 1).float()
                loss = criterion(output, target)
                total_loss += float(loss.item())
                num_batches += 1
        return total_loss / max(num_batches, 1)

    def fit(
        self,
        data: MolFrame,
        lr: float = 1e-4,
        eps: int = 20,
        verbose: bool = False,
        prior: float = 0.75,
        nnPU: bool = True,
        batch_size: int = 64,
        weight_decay: float = 1e-6,
        val_data: Optional[MolFrame] = None,
        patience: int = 7,
        min_delta: float = 1e-4,
        timeout_seconds: Optional[float] = None,
    ) -> "Filter":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = PULoss(prior) if nnPU else nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if self.mode == "pair":
            loader = self._pair_loader(data, batch_size=batch_size)
        else:
            loader = self._single_loader(data, batch_size=batch_size)

        val_loader = None
        if val_data is not None:
            if self.mode == "pair":
                val_loader = self._pair_loader(val_data, batch_size=batch_size, shuffle=False)
            else:
                val_loader = self._single_loader(val_data, batch_size=batch_size, shuffle=False)

        best_loss = float("inf")
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        loss_history: List[float] = []
        val_loss_history: List[float] = []
        early_stopped_epoch: Optional[int] = None
        timed_out = False
        stop_reason = "completed"
        last_epoch_seconds: Optional[float] = None
        started = time.perf_counter()

        def _sync_training_state() -> None:
            self.best_state_ = best_state
            self.best_loss_ = best_loss if loss_history else None
            self.best_val_loss_ = best_val_loss if val_loader is not None and val_loss_history else None
            self.loss_history_ = list(loss_history)
            self.val_loss_history_ = list(val_loss_history)
            self.epochs_trained_ = len(loss_history)
            self.early_stopped_epoch_ = early_stopped_epoch
            self.timed_out_ = timed_out
            self.timeout_seconds_ = timeout_seconds
            self.stop_reason_ = stop_reason
            self.last_epoch_seconds_ = last_epoch_seconds

        _sync_training_state()
        for epoch in range(eps):
            if timeout_seconds is not None and last_epoch_seconds is not None:
                elapsed = time.perf_counter() - started
                remaining = timeout_seconds - elapsed
                required_seconds = last_epoch_seconds + _timeout_buffer(last_epoch_seconds)
                if remaining <= required_seconds:
                    timed_out = True
                    stop_reason = "timeout"
                    if verbose:
                        print(
                            f"filter stopping before epoch={epoch + 1} "
                            f"remaining={remaining:.1f}s estimated_epoch={last_epoch_seconds:.1f}s"
                        )
                    break
            epoch_started = time.perf_counter()
            self.train()
            epoch_loss = 0.0
            num_batches = 0
            for batch in loader:
                optimizer.zero_grad()
                if self.mode == "pair":
                    batch = batch.to(device)
                    output = self(batch)
                    target = batch.y.view(-1, 1).float()
                else:
                    sub_batch, prod_batch = batch
                    sub_batch = sub_batch.to(device)
                    prod_batch = prod_batch.to(device)
                    output = self(sub_batch, prod_batch)
                    target = prod_batch.y.view(-1, 1).float()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            if num_batches == 0:
                stop_reason = "no_batches"
                break
            avg_loss = epoch_loss / num_batches
            loss_history.append(float(avg_loss))
            best_loss = min(best_loss, avg_loss)
            if val_loader is not None:
                val_loss = self._compute_loader_loss(val_loader, criterion, device)
                val_loss_history.append(float(val_loss))
                if verbose:
                    print(f"filter epoch={epoch + 1} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
                    self.best_state_ = best_state
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    early_stopped_epoch = epoch + 1
                    stop_reason = "early_stopping"
                    if verbose:
                        print(f"filter early stopping at epoch={epoch + 1} patience={patience}")
                    break
            elif verbose:
                print(f"filter epoch={epoch + 1} loss={avg_loss:.4f}")
            last_epoch_seconds = time.perf_counter() - epoch_started
            _sync_training_state()
            if timeout_seconds is not None and (time.perf_counter() - started) >= timeout_seconds:
                timed_out = True
                stop_reason = "timeout"
                _sync_training_state()
                if verbose:
                    print(f"filter stopping at epoch={epoch + 1} timeout_seconds={timeout_seconds}")
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        _sync_training_state()
        return self

    @torch.no_grad()
    def calibrate_threshold(
        self,
        val_data: MolFrame,
        target: Literal["f1", "mcc"] = "f1",
        verbose: bool = True,
    ) -> Tuple[float, float]:
        all_scores: List[float] = []
        all_labels: List[float] = []

        self.eval()
        for substrate, true_products in val_data.map.items():
            product_pool = set(true_products) | set(val_data.gen_map.get(substrate, set()))
            for product in product_pool:
                try:
                    score = float(self.score(substrate, product))
                except Exception:
                    continue
                all_scores.append(score)
                all_labels.append(1.0 if product in true_products else 0.0)

        if not all_scores:
            self.calibrated_threshold = 0.5
            return 0.5, 0.0

        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        labels_t = torch.tensor(all_labels, dtype=torch.float32)
        best_threshold = 0.5
        best_metric = -1.0

        for threshold_step in range(1, 100):
            threshold = threshold_step / 100.0
            predictions = (scores_t >= threshold).float()
            tp = float((predictions * labels_t).sum().item())
            fp = float((predictions * (1.0 - labels_t)).sum().item())
            fn = float((((1.0 - predictions) * labels_t)).sum().item())
            tn = float((((1.0 - predictions) * (1.0 - labels_t))).sum().item())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
            mcc_num = tp * tn - fp * fn
            mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + 1e-8
            mcc = mcc_num / mcc_den
            metric = f1 if target == "f1" else mcc
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold

        self.calibrated_threshold = best_threshold
        if verbose:
            print(
                f"filter threshold calibrated={best_threshold:.3f} "
                f"target={target} metric={best_metric:.4f}"
            )
        return best_threshold, best_metric

    @torch.no_grad()
    def score(self, sub: str, prod: str, pca: bool = False) -> float:
        del pca
        sub_mol = Chem.MolFromSmiles(sub)
        prod_mol = Chem.MolFromSmiles(prod)
        if sub_mol is None or prod_mol is None:
            return 0.0

        first_param = next(iter(self.parameters()), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        self.eval()
        if self.mode == "pair":
            graph = build_pair_graph(sub_mol, prod_mol)
            if graph is None:
                return 0.0
            batch = Batch.from_data_list([graph]).to(device)
            return float(self(batch).view(-1).item())

        sub_graph = from_rdmol(sub_mol)
        prod_graph = from_rdmol(prod_mol)
        if sub_graph is None or prod_graph is None:
            return 0.0
        sub_batch = Batch.from_data_list([sub_graph]).to(device)
        prod_batch = Batch.from_data_list([prod_graph]).to(device)
        return float(self(sub_batch, prod_batch).view(-1).item())

    @torch.no_grad()
    def predict(self, sub: str, prod: str, pca: bool = False, threshold: Optional[float] = None) -> int:
        cutoff = float(self.calibrated_threshold if threshold is None else threshold)
        return int(self.score(sub, prod, pca=pca) >= cutoff)


class GATv2Filter(Filter):
    def __init__(self, in_channels: int, edge_dim: int, arg_vec: Sequence[int], mode: Literal["single", "pair"], **kwargs) -> None:
        super().__init__(in_channels, edge_dim, arg_vec, mode=mode, conv_kind="gatv2", **kwargs)


class GCNFilter(Filter):
    def __init__(self, in_channels: int, edge_dim: int, arg_vec: Sequence[int], mode: Literal["single", "pair"], **kwargs) -> None:
        super().__init__(in_channels, edge_dim, arg_vec, mode=mode, conv_kind="gcn", **kwargs)


class GINFilter(Filter):
    def __init__(self, in_channels: int, edge_dim: int, arg_vec: Sequence[int], mode: Literal["single", "pair"], **kwargs) -> None:
        super().__init__(in_channels, edge_dim, arg_vec, mode=mode, conv_kind="gin", **kwargs)


class MolPathFilter(Filter):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        arg_vec: Sequence[int],
        mode: Literal["single", "pair"],
        molpath_cutoff: Optional[int] = None,
        molpath_y: Optional[float] = None,
        molpath_hidden: Optional[int] = None,
        **kwargs,
    ) -> None:
        del molpath_cutoff, molpath_y, molpath_hidden
        super().__init__(in_channels, edge_dim, arg_vec, mode=mode, conv_kind="gatv2", **kwargs)


class MorganOnlyFilter(Filter):
    def __init__(self, in_channels: int, edge_dim: int, arg_vec: Sequence[int], mode: Literal["single", "pair"]) -> None:
        del in_channels, edge_dim
        hidden = _pad_dims(arg_vec, 4, 256)
        GFilter.__init__(self)
        self.mode = mode
        input_dim = 2 * FINGERPRINT_DIM
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, data: Data, met: Optional[Data] = None) -> torch.Tensor:
        if self.mode == "pair":
            fp = data.fp.float()
        else:
            if met is None:
                raise ValueError("met is required in single mode")
            fp = torch.cat((data.fp.float(), met.fp.float()), dim=1)
        return torch.sigmoid(self.classifier(fp))

    def fit(
        self,
        data: MolFrame,
        lr: float = 1e-4,
        eps: int = 20,
        verbose: bool = False,
        prior: float = 0.75,
        nnPU: bool = True,
        batch_size: int = 64,
        weight_decay: float = 1e-6,
        val_data: Optional[MolFrame] = None,
        patience: int = 7,
        min_delta: float = 1e-4,
        timeout_seconds: Optional[float] = None,
    ) -> "MorganOnlyFilter":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = PULoss(prior) if nnPU else nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if self.mode == "pair":
            loader = self._pair_loader(data, batch_size=batch_size, shuffle=True)
            if val_data is not None:
                val_loader = self._pair_loader(val_data, batch_size=batch_size, shuffle=False)
            else:
                val_loader = None
        else:
            data.singlegraphs()

            def collate(items: List[Tuple[Data, Data]]) -> Tuple[Batch, Batch]:
                return Batch.from_data_list([item[0] for item in items]), Batch.from_data_list([item[1] for item in items])

            loader = DataLoader(_SingleDataset(data._single_dataset()), batch_size=batch_size, shuffle=True, collate_fn=collate)
            if val_data is not None:
                val_data.singlegraphs()
                val_loader = DataLoader(_SingleDataset(val_data._single_dataset()), batch_size=batch_size, shuffle=False, collate_fn=collate)
            else:
                val_loader = None

        best_loss = float("inf")
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        loss_history: List[float] = []
        val_loss_history: List[float] = []
        early_stopped_epoch: Optional[int] = None
        timed_out = False
        stop_reason = "completed"
        last_epoch_seconds: Optional[float] = None
        started = time.perf_counter()

        def _sync_training_state() -> None:
            self.best_state_ = best_state
            self.best_loss_ = best_loss if loss_history else None
            self.best_val_loss_ = best_val_loss if val_loader is not None and val_loss_history else None
            self.loss_history_ = list(loss_history)
            self.val_loss_history_ = list(val_loss_history)
            self.epochs_trained_ = len(loss_history)
            self.early_stopped_epoch_ = early_stopped_epoch
            self.timed_out_ = timed_out
            self.timeout_seconds_ = timeout_seconds
            self.stop_reason_ = stop_reason
            self.last_epoch_seconds_ = last_epoch_seconds

        _sync_training_state()
        for epoch in range(eps):
            if timeout_seconds is not None and last_epoch_seconds is not None:
                elapsed = time.perf_counter() - started
                remaining = timeout_seconds - elapsed
                required_seconds = last_epoch_seconds + _timeout_buffer(last_epoch_seconds)
                if remaining <= required_seconds:
                    timed_out = True
                    stop_reason = "timeout"
                    if verbose:
                        print(
                            f"morgan filter stopping before epoch={epoch + 1} "
                            f"remaining={remaining:.1f}s estimated_epoch={last_epoch_seconds:.1f}s"
                        )
                    break
            epoch_started = time.perf_counter()
            self.train()
            epoch_loss = 0.0
            num_batches = 0
            for batch in loader:
                optimizer.zero_grad()
                if self.mode == "pair":
                    batch = batch.to(device)
                    output = self(batch)
                    target = batch.y.view(-1, 1).float()
                else:
                    sub_batch, prod_batch = batch
                    sub_batch = sub_batch.to(device)
                    prod_batch = prod_batch.to(device)
                    output = self(sub_batch, prod_batch)
                    target = prod_batch.y.view(-1, 1).float()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            if not num_batches:
                stop_reason = "no_batches"
                break
            avg_loss = epoch_loss / num_batches
            loss_history.append(float(avg_loss))
            best_loss = min(best_loss, avg_loss)
            if val_loader is not None:
                val_loss = self._compute_loader_loss(val_loader, criterion, device)
                val_loss_history.append(float(val_loss))
                if verbose:
                    print(f"morgan filter epoch={epoch + 1} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
                    self.best_state_ = best_state
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    early_stopped_epoch = epoch + 1
                    stop_reason = "early_stopping"
                    if verbose:
                        print(f"morgan filter early stopping at epoch={epoch + 1} patience={patience}")
                    break
            elif verbose:
                print(f"morgan filter epoch={epoch + 1} loss={avg_loss:.4f}")
            last_epoch_seconds = time.perf_counter() - epoch_started
            _sync_training_state()
            if timeout_seconds is not None and (time.perf_counter() - started) >= timeout_seconds:
                timed_out = True
                stop_reason = "timeout"
                _sync_training_state()
                if verbose:
                    print(f"morgan filter stopping at epoch={epoch + 1} timeout_seconds={timeout_seconds}")
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        _sync_training_state()
        return self

    @torch.no_grad()
    def score(self, sub: str, prod: str, pca: bool = False) -> float:
        del pca
        sub_mol = Chem.MolFromSmiles(sub)
        prod_mol = Chem.MolFromSmiles(prod)
        if sub_mol is None or prod_mol is None:
            return 0.0
        first_param = next(iter(self.parameters()), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        if self.mode == "pair":
            graph = build_pair_graph(sub_mol, prod_mol)
            if graph is None:
                return 0.0
            batch = Batch.from_data_list([graph]).to(device)
            return float(self(batch).view(-1).item())
        sub_graph = from_rdmol(sub_mol)
        prod_graph = from_rdmol(prod_mol)
        if sub_graph is None or prod_graph is None:
            return 0.0
        sub_batch = Batch.from_data_list([sub_graph]).to(device)
        prod_batch = Batch.from_data_list([prod_graph]).to(device)
        return float(self(sub_batch, prod_batch).view(-1).item())

    @torch.no_grad()
    def predict(self, sub: str, prod: str, pca: bool = False, threshold: Optional[float] = None) -> int:
        cutoff = float(self.calibrated_threshold if threshold is None else threshold)
        return int(self.score(sub, prod, pca=pca) >= cutoff)
