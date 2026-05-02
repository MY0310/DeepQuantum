"""
Topological invariance verification with real Q-GAD model (standardized).
"""

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path


def _patch_windows_conda_dll_path() -> None:
    if os.name != "nt":
        return
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        exe_parent = Path(sys.executable).resolve().parent
        if (exe_parent / "conda-meta").exists():
            prefix = str(exe_parent)
    if not prefix:
        return

    dll_dirs = [
        Path(prefix),
        Path(prefix) / "Library" / "mingw-w64" / "bin",
        Path(prefix) / "Library" / "usr" / "bin",
        Path(prefix) / "Library" / "bin",
        Path(prefix) / "Scripts",
    ]
    existing = [str(p) for p in dll_dirs if p.exists()]
    if not existing:
        return

    path_parts = [p for p in os.environ.get("PATH", "").split(";") if p]
    for p in reversed(existing):
        if p not in path_parts:
            path_parts.insert(0, p)
    os.environ["PATH"] = ";".join(path_parts)

    add_dll = getattr(os, "add_dll_directory", None)
    if add_dll is not None:
        for p in existing:
            try:
                add_dll(p)
            except OSError:
                pass


_patch_windows_conda_dll_path()

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.helpers import get_device, load_qgad_checkpoint_model, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topological Invariance with Q-GAD (Standardized)")
    parser.add_argument("--n-pairs", type=int, default=20)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--similarity-method", type=str, default="cosine", choices=["cosine", "euclidean", "correlation"])
    parser.add_argument(
        "--feature-normalization",
        type=str,
        default="zscore",
        choices=["none", "zscore", "robust"],
        help="Normalize quantum features before pairwise similarity.",
    )
    parser.add_argument("--n-shots", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show-quantum-logs", action="store_true")
    parser.add_argument(
        "--canonicalize-graph",
        action="store_true",
        default=False,
        help="Apply deterministic canonical-like node ordering before encoding.",
    )
    parser.add_argument(
        "--no-canonicalize-graph",
        dest="canonicalize_graph",
        action="store_false",
        help="Disable canonical-like node ordering before encoding.",
    )
    parser.add_argument("--output-dir", default="experiments/topological_invariance/results")
    return parser.parse_args()


def _forward_quantum(quantum_extractor, squeezing, unitary, quiet_quantum_logs=True):
    if not quiet_quantum_logs:
        return quantum_extractor(squeezing, unitary)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return quantum_extractor(squeezing, unitary)


def load_qgad_quantum_extractor(device: str, n_shots: int):
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "elliptic_model.pt"
    model, checkpoint, inferred = load_qgad_checkpoint_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        n_modes=20,
        n_shots=n_shots,
    )

    has_real_backend = bool(getattr(model.quantum_extractor.gbs_kernel, "has_deepquantum", False))
    best_f1 = max(checkpoint.get("history", {}).get("val_f1", [0.0]))
    print(f"[Model] Loaded {checkpoint_path.name} (best val_f1={best_f1:.4f}, n_shots={n_shots})")
    print(f"[Model] Inferred architecture hidden_dims={inferred['hidden_dims']}")
    print(f"[Model] Real DeepQuantum backend: {has_real_backend}")
    if not has_real_backend:
        raise RuntimeError("DeepQuantum backend unavailable. Mock backend is not allowed.")
    return model.quantum_extractor


def encode_graph_to_params(G, n_modes=20, max_squeezing=2.0):
    G = G.copy()
    n_nodes = G.number_of_nodes()
    if n_nodes < n_modes:
        for i in range(n_modes - n_nodes):
            G.add_node(n_nodes + i)

    adj_matrix = nx.to_numpy_array(G, nodelist=range(n_modes))
    degrees = np.array([G.degree(i) for i in range(n_modes)], dtype=np.float32)
    max_degree = np.max(degrees) if np.max(degrees) > 0 else 1.0
    squeezing = (degrees / max_degree) * max_squeezing

    try:
        u, _, vt = np.linalg.svd(adj_matrix)
        unitary = u @ vt
    except np.linalg.LinAlgError:
        h = np.random.randn(n_modes, n_modes)
        q, _ = np.linalg.qr(h)
        unitary = q

    return squeezing.astype(np.float32), unitary.astype(np.float32)


def _wl_refinement_scores(G, rounds=4):
    labels = {node: str(G.degree(node)) for node in G.nodes()}
    for _ in range(rounds):
        new_labels = {}
        for node in G.nodes():
            neigh = sorted(labels[nbr] for nbr in G.neighbors(node))
            signature = labels[node] + "|" + ",".join(neigh)
            new_labels[node] = str(hash(signature))
        labels = new_labels
    return labels


def canonicalize_graph_for_encoding(G):
    G = G.copy()
    scores = _wl_refinement_scores(G, rounds=4)
    degree = dict(G.degree())
    clustering = nx.clustering(G)
    # Deterministic ordering with structural keys first, raw node id as final tiebreaker.
    ordered_nodes = sorted(
        G.nodes(),
        key=lambda n: (scores[n], degree[n], clustering[n], str(n)),
    )
    mapping = {node: idx for idx, node in enumerate(ordered_nodes)}
    return nx.relabel_nodes(G, mapping)


def generate_graph_pair(n_nodes=20, seed=42, isomorphic=True):
    rng = np.random.default_rng(seed)
    if isomorphic:
        g1 = nx.erdos_renyi_graph(n_nodes, 0.3, seed=seed)
        permutation = rng.permutation(n_nodes)
        mapping = {i: int(permutation[i]) for i in range(n_nodes)}
        g2 = nx.relabel_nodes(g1, mapping)
    else:
        # Use structurally distinct families to make non-isomorphic pairs
        # less likely to collapse in similarity space.
        g1 = nx.barabasi_albert_graph(n_nodes, 2, seed=seed)
        g2 = nx.watts_strogatz_graph(n_nodes, 4, 0.6, seed=seed + 1)
    return g1, g2


@torch.no_grad()
def extract_quantum_features(
    graph,
    quantum_extractor,
    device,
    quiet_quantum_logs=True,
    canonicalize_graph=False,
):
    gbs_kernel = quantum_extractor.gbs_kernel
    graph_for_encoding = canonicalize_graph_for_encoding(graph) if canonicalize_graph else graph
    squeezing, unitary = encode_graph_to_params(
        graph_for_encoding,
        n_modes=gbs_kernel.n_modes,
        max_squeezing=gbs_kernel.config.max_squeezing,
    )
    squeezing = torch.tensor(squeezing, dtype=torch.float32).unsqueeze(0)
    unitary = torch.tensor(unitary, dtype=torch.float32).unsqueeze(0)
    # Correct path: directly use quantum_extractor output (already quantum features).
    features = _forward_quantum(quantum_extractor, squeezing, unitary, quiet_quantum_logs=quiet_quantum_logs)
    return features.squeeze(0).detach().cpu().numpy()


def compute_similarity(features1, features2, method="cosine") -> float:
    features1 = np.nan_to_num(features1, nan=0.0).reshape(1, -1)
    features2 = np.nan_to_num(features2, nan=0.0).reshape(1, -1)

    if method == "cosine":
        n1, n2 = np.linalg.norm(features1), np.linalg.norm(features2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(cosine_similarity(features1, features2)[0, 0])
    if method == "euclidean":
        distance = float(np.linalg.norm(features1 - features2))
        return float(1.0 / (1.0 + distance))
    if method == "correlation":
        corr = float(np.corrcoef(features1.flatten(), features2.flatten())[0, 1])
        corr = 0.0 if np.isnan(corr) else corr
        return float((corr + 1.0) / 2.0)
    raise ValueError(f"Unknown similarity method: {method}")


def normalize_features(feature_list, method="zscore"):
    if method == "none":
        return [np.asarray(f, dtype=np.float32) for f in feature_list]

    stacked = np.stack([np.asarray(f, dtype=np.float32) for f in feature_list], axis=0)
    eps = 1e-8

    if method == "zscore":
        center = np.mean(stacked, axis=0, keepdims=True)
        scale = np.std(stacked, axis=0, keepdims=True)
    elif method == "robust":
        center = np.median(stacked, axis=0, keepdims=True)
        scale = np.median(np.abs(stacked - center), axis=0, keepdims=True) * 1.4826
    else:
        raise ValueError(f"Unknown feature normalization: {method}")

    scale = np.where(scale < eps, 1.0, scale)
    normalized = (stacked - center) / scale
    return [normalized[i] for i in range(normalized.shape[0])]


def run(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device(args.device)
    quiet_quantum_logs = not args.show_quantum_logs

    print("\n" + "=" * 80)
    print("Topological Invariance Verification (Standardized)")
    print("=" * 80)
    print(
        f"[Config] device={device}, n_pairs={args.n_pairs}, n_nodes={args.n_nodes}, "
        f"similarity={args.similarity_method}, normalize={args.feature_normalization}, "
        f"canonicalize={args.canonicalize_graph}, n_shots={args.n_shots}"
    )

    quantum_extractor = load_qgad_quantum_extractor(device=device, n_shots=args.n_shots)

    iso_pair_features = []
    non_iso_pair_features = []

    for i in range(args.n_pairs):
        g1, g2 = generate_graph_pair(n_nodes=args.n_nodes, seed=args.seed + i, isomorphic=True)
        f1 = extract_quantum_features(
            g1,
            quantum_extractor,
            device,
            quiet_quantum_logs=quiet_quantum_logs,
            canonicalize_graph=args.canonicalize_graph,
        )
        f2 = extract_quantum_features(
            g2,
            quantum_extractor,
            device,
            quiet_quantum_logs=quiet_quantum_logs,
            canonicalize_graph=args.canonicalize_graph,
        )
        iso_pair_features.append((f1, f2))

    for i in range(args.n_pairs):
        g1, g2 = generate_graph_pair(n_nodes=args.n_nodes, seed=args.seed + 1000 + i, isomorphic=False)
        f1 = extract_quantum_features(
            g1,
            quantum_extractor,
            device,
            quiet_quantum_logs=quiet_quantum_logs,
            canonicalize_graph=args.canonicalize_graph,
        )
        f2 = extract_quantum_features(
            g2,
            quantum_extractor,
            device,
            quiet_quantum_logs=quiet_quantum_logs,
            canonicalize_graph=args.canonicalize_graph,
        )
        non_iso_pair_features.append((f1, f2))

    all_features = [f for pair in (iso_pair_features + non_iso_pair_features) for f in pair]
    normalized_features = normalize_features(all_features, method=args.feature_normalization)

    idx = 0
    norm_iso_pairs = []
    for _ in iso_pair_features:
        norm_iso_pairs.append((normalized_features[idx], normalized_features[idx + 1]))
        idx += 2

    norm_non_iso_pairs = []
    for _ in non_iso_pair_features:
        norm_non_iso_pairs.append((normalized_features[idx], normalized_features[idx + 1]))
        idx += 2

    iso_scores = [compute_similarity(f1, f2, method=args.similarity_method) for f1, f2 in norm_iso_pairs]
    non_iso_scores = [compute_similarity(f1, f2, method=args.similarity_method) for f1, f2 in norm_non_iso_pairs]

    iso_mean, iso_std = float(np.mean(iso_scores)), float(np.std(iso_scores))
    non_iso_mean, non_iso_std = float(np.mean(non_iso_scores)), float(np.std(non_iso_scores))
    separation = iso_mean - non_iso_mean
    passed = bool(separation > 0)

    print(f"[Result] iso={iso_mean:.4f}±{iso_std:.4f}, non_iso={non_iso_mean:.4f}±{non_iso_std:.4f}, sep={separation:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": "topological_invariance_qgad_standardized",
        "parameters": {
            "seed": int(args.seed),
            "n_pairs": int(args.n_pairs),
            "n_nodes": int(args.n_nodes),
            "similarity_method": args.similarity_method,
            "feature_normalization": args.feature_normalization,
            "canonicalize_graph": bool(args.canonicalize_graph),
            "n_shots": int(args.n_shots),
            "device": str(device),
            "has_deepquantum": True,
        },
        "isomorphic_pairs": {
            "mean_similarity": iso_mean,
            "std_similarity": iso_std,
            "similarities": [float(x) for x in iso_scores],
        },
        "non_isomorphic_pairs": {
            "mean_similarity": non_iso_mean,
            "std_similarity": non_iso_std,
            "similarities": [float(x) for x in non_iso_scores],
        },
        "verification": {"separation": float(separation), "passed": passed},
    }

    json_path = output_dir / "topological_invariance_qgad_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "topological_invariance_qgad_comparison.csv"
    pd.DataFrame(
        {
            "Graph Type": ["Isomorphic", "Non-Isomorphic"],
            "Mean Similarity": [iso_mean, non_iso_mean],
            "Std Similarity": [iso_std, non_iso_std],
            "Count": [args.n_pairs, args.n_pairs],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[Saved] {json_path}")
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    run(parse_args())
