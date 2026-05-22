"""Point d'entrée d'entraînement GNN pour le projet MIT-BIH.

Mode unique:
- gnn: entraînement d'un GNN de graphes de battements (GraphSAGE)

Le script bascule automatiquement sur des données synthétiques si le dataset
MIT-BIH n'est pas encore téléchargé.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.data.preprocessing import segment_beats, summarize_segment, build_graph_sample
from src.data.mitbih_loader import list_records, load_record, load_annotations, extract_annotated_beats


PYG_AVAILABLE = None


def _pyg_available() -> bool:
    global PYG_AVAILABLE
    if PYG_AVAILABLE is not None:
        return PYG_AVAILABLE
    try:
        import torch_geometric  # noqa: F401
        PYG_AVAILABLE = True
    except Exception:
        PYG_AVAILABLE = False
    return PYG_AVAILABLE


LABEL_TO_ID = {'normal': 0, 'arrhythmia': 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


@dataclass
class DatasetBundle:
    graphs: List[Any]
    features: np.ndarray
    labels: np.ndarray


def _seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # L'entraînement GNN requiert torch; on laisse l'erreur apparaître plus tard.
        pass


def _synthetic_bundle(
    num_graphs: int = 64,
    num_nodes: int = 12,
    num_features: int = 10,
    need_graphs: bool = True,
) -> DatasetBundle:
    graphs: List["Data"] = []
    features = []
    labels = []
    for i in range(num_graphs):
        features.append(np.random.randn(10))
        labels.append(i % 2)
        if need_graphs:
            try:
                import torch
                from torch_geometric.data import Data
            except Exception as e:
                raise RuntimeError('Le mode GNN nécessite PyTorch et torch-geometric fonctionnels.') from e

            x = torch.randn(num_nodes, num_features, dtype=torch.float32)
            edges = []
            for j in range(num_nodes - 1):
                edges.append((j, j + 1))
                edges.append((j + 1, j))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            y = torch.tensor([i % 2], dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return DatasetBundle(graphs=graphs, features=np.asarray(features), labels=np.asarray(labels))


def _build_segments_from_record(record_name: str, data_dir: str, window_size: int, stride: int) -> Tuple[List[List[dict]], int]:
    record = load_record(record_name, data_dir)
    ann = load_annotations(record_name, data_dir)
    beats = extract_annotated_beats(record['signal'], record['fs'], ann['sample'], ann['symbol'])
    segments = segment_beats(beats, window_size=window_size, stride=stride)
    return segments, int(record['fs'])


def build_mitbih_bundle(
    data_dir: str = 'data/mitbih',
    max_records: int = 0,
    window_size: int = 12,
    stride: int = 6,
    need_graphs: bool = True,
) -> DatasetBundle:
    if need_graphs and not _pyg_available():
        raise ImportError('torch-geometric est requis pour le mode GNN')

    records = list_records(data_dir)
    if not records:
        print(f'No MIT-BIH records found in {data_dir}; using synthetic fallback.')
        return _synthetic_bundle(need_graphs=need_graphs)

    graphs: List["Data"] = []
    feature_rows: List[np.ndarray] = []
    labels: List[int] = []

    selected_records = records if max_records is None or max_records <= 0 else records[:max_records]

    for record_name in selected_records:
        try:
            segments, fs = _build_segments_from_record(record_name, data_dir, window_size, stride)
        except Exception as e:
            print(f'Skipping {record_name}: {e}')
            continue

        for segment in segments:
            summary = summarize_segment(segment, fs)
            feature_rows.append(np.array([
                summary['num_beats'],
                summary['mean_rr_prev'],
                summary['mean_rr_next'],
                summary['mean_amp'],
                summary['std_amp'],
            ], dtype=np.float32))
            label = LABEL_TO_ID[summary['label']]
            labels.append(label)

            if need_graphs:
                import torch
                from torch_geometric.data import Data

                graph_dict = build_graph_sample(segment, fs, k=1)
                x = torch.tensor(graph_dict['x'], dtype=torch.float32)
                edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
                y = torch.tensor([label], dtype=torch.long)
                graphs.append(Data(x=x, edge_index=edge_index, y=y))

    if not graphs:
        print('No usable MIT-BIH segments found; using synthetic fallback.')
        return _synthetic_bundle(need_graphs=need_graphs)

    return DatasetBundle(graphs=graphs, features=np.asarray(feature_rows), labels=np.asarray(labels))


def _gnn_model(in_dim: int):
    try:
        from src.models.gnn_models import GraphSAGEModel
    except Exception as e:
        raise RuntimeError('Impossible d import the GNN models because torch is not functional.') from e
    return GraphSAGEModel(in_dim=in_dim, hidden=128, num_layers=2, out_dim=128)


def train_gnn(bundle: DatasetBundle, epochs: int = 10, batch_size: int = 16, seed: int = 42) -> None:
    if not _pyg_available():
        raise ImportError('torch-geometric est requis pour le mode GNN')

    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError(
            'PyTorch ne démarre pas dans cet environnement. Le mode GNN ne peut pas s exécuter tant que torch n est pas réparé.'
        ) from e

    from sklearn.model_selection import train_test_split

    if len(bundle.graphs) < 4:
        raise ValueError('Too few graphs for meaningful GNN training (need at least 4 graphs).')

    graphs = bundle.graphs
    labels = np.asarray([int(g.y.item()) for g in graphs], dtype=int)
    train_idx, test_idx = train_test_split(np.arange(len(graphs)), test_size=0.25, random_state=seed, stratify=labels)
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    from torch_geometric.loader import DataLoader as PyGDataLoader

    train_loader = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _gnn_model(in_dim=graphs[0].num_node_features).to(device)
    classifier = nn.Linear(128, len(LABEL_TO_ID)).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    from sklearn.metrics import accuracy_score, f1_score, classification_report

    # tracking best model by validation f1_macro
    best_metric = float('-inf')
    best_path = os.path.join('weights', 'gnn_best.pth')
    os.makedirs('weights', exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        classifier.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            emb = model(batch.x, batch.edge_index, batch.batch)
            logits = classifier(emb)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs
        avg_loss = total_loss / max(1, len(train_graphs))
        # validation pass (compute f1 on test set)
        model.eval()
        classifier.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                emb = model(batch.x, batch.edge_index, batch.batch)
                logits = classifier(emb)
                preds = torch.argmax(logits, dim=-1)
                y_true.extend(batch.y.view(-1).cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average='macro')
        print(f'Epoch {epoch:02d} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1_macro={val_f1:.4f}')

        # save best
        if val_f1 > best_metric:
            best_metric = val_f1
            ck = {
                'model_state': model.state_dict(),
                'classifier_state': classifier.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
            }
            try:
                torch.save(ck, best_path)
                print('Saved new best checkpoint:', best_path)
            except Exception as e:
                print('Warning: failed to save checkpoint:', e)

    # final evaluation printout (reload best if available)
    if os.path.isfile(best_path):
        try:
            ck = torch.load(best_path, map_location=device)
            model.load_state_dict(ck['model_state'])
            classifier.load_state_dict(ck['classifier_state'])
            print('Loaded best checkpoint from', best_path, 'epoch', ck.get('epoch'))
        except Exception:
            print('Could not load best checkpoint for final eval; using current model')

    model.eval()
    classifier.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            emb = model(batch.x, batch.edge_index, batch.batch)
            logits = classifier(emb)
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(batch.y.view(-1).cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    print('Final GNN accuracy:', accuracy_score(y_true, y_pred))
    print('Final GNN f1_macro:', f1_score(y_true, y_pred, average='macro'))
    print(classification_report(y_true, y_pred, target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='MIT-BIH + GNN training entrypoint')
    parser.add_argument('--mode', choices=['gnn'], default='gnn')
    parser.add_argument('--data-dir', type=str, default='data/mitbih')
    parser.add_argument('--max-records', type=int, default=0, help='0 = all records')
    parser.add_argument('--window-size', type=int, default=12)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _seed_everything(args.seed)

    bundle = build_mitbih_bundle(
        data_dir=args.data_dir,
        max_records=args.max_records,
        window_size=args.window_size,
        stride=args.stride,
        need_graphs=True,
    )
    train_gnn(bundle, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)


if __name__ == '__main__':
    main()
