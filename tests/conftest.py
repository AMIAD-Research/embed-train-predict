import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
# the Hydra config of the embed phase ships inside the package, next to the
# module that composes it
CONFIG_DIR = REPO / "etp" / "cli" / "config"
# allow running the tests without installing the package
sys.path.insert(0, str(REPO))
TESTS = Path(__file__).resolve().parent
# make tests/stubs.py importable by Hydra targets
sys.path.insert(0, str(TESTS))
# scripts/ holds no package, so put it on the path here rather than leaving
# test_prepare_data.py to import prepare_clc_fce_data halfway down the module
sys.path.insert(0, str(REPO / "scripts"))


def run_cli(*argv, cwd=None):
    """Run the `etp` command line in a subprocess and return the CompletedProcess.

    Calls etp.cli.main:cli directly, the function the `etp` console script points
    at, so the tests exercise the same entry point whether or not the package is
    installed in the environment.
    """
    return subprocess.run(
        [sys.executable, "-c", "from etp.cli.main import cli; cli()", *argv],
        cwd=str(cwd or REPO),
        # the caller asserts on returncode, a failing phase included, so a
        # non-zero exit is a result to hand back and not something to raise on
        check=False,
        # tests/ on the path too, so a Hydra target can name a stub from stubs.py
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(TESTS)])},
        capture_output=True,
        text=True,
    )


@pytest.fixture
def corpus_dir(tmp_path):
    """Synthetic corpus (list + annotation csv): 4 rows per split, 3 listed."""
    root = tmp_path / "corpus"
    for split, n0 in [("train", 0), ("validation", 10), ("test", 20)]:
        d = root / "clc-fce"
        (d / "list").mkdir(parents=True, exist_ok=True)
        (d / "annotation").mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "id": f"clc-fce-{split}-{i}",
                "language_l1": lang,
                "text": f"some {split} text number {i} written here",
                "extra": "x",
            }
            for i, lang in zip(range(n0, n0 + 4), ["fra", "deu", "spa", "ita"])
        ]
        ann_file = f"annotation/{split}.csv"
        pd.DataFrame(rows).to_csv(d / ann_file, index=False)
        pd.DataFrame(
            {"id": [r["id"] for r in rows[:-1]], "annotation_file": ann_file}
        ).to_csv(d / "list" / f"clc-fce-{split}.csv", index=False)
    return root


def _write_embeddings(path, n=40, dim=4, layers=1):
    """Write n linearly separable dim-d embeddings for two classes to `path` (h5).

    `layers` datasets are stored, layer_0 .. layer_<layers-1>, so a test can ask
    for `--layer all` or `--layer last` and get more than one answer.

    Returns (ids, labels), the first 30 meant as the train split and the rest as
    the test split.
    """
    import numpy as np
    import torch

    from etp.embedder.writers import WriterHDF5Embedding

    rng = np.random.default_rng(0)
    ids = [f"s{i:02d}" for i in range(n)]
    labels = ["a" if i % 2 == 0 else "b" for i in range(n)]
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X[:, 0] += np.where(np.array(labels) == "a", 4.0, -4.0)  # linearly separable

    writer = WriterHDF5Embedding(file_name=str(path))
    writer.write(
        ids, {f"layer_{i}": torch.tensor(X) * (1.0 + i) for i in range(layers)}
    )
    writer.close()

    return ids, labels


@pytest.fixture
def embeddings_file(tmp_path):
    """The h5 path plus the train / test label tables (30 / 10 rows).

    Shared by the phase 2 (train) and phase 3 (predict) tests.
    """
    path = tmp_path / "embeddings.h5"
    ids, labels = _write_embeddings(path)

    train = pd.DataFrame({"id": ids[:30], "label": labels[:30]})
    test = pd.DataFrame({"id": ids[30:], "label": labels[30:]})
    return str(path), train, test


@pytest.fixture
def pipeline_run(tmp_path):
    """A populated run-dir plus a matching corpus, for the `etp train` / `etp predict` phases.

    Writes <run-dir>/embeddings.h5 with two layers and a 2-class corpus whose
    annotation ids line up with the embeddings, split into train (30) / test (10)
    lists. Returns (run_dir, corpus_dir) as Paths, ready to be passed to the
    command line.
    """
    run_dir = tmp_path / "out" / "run"
    ids, labels = _write_embeddings(run_dir / "embeddings.h5", layers=2)

    corpus = tmp_path / "corpus" / "example"
    (corpus / "list").mkdir(parents=True)
    (corpus / "annotation").mkdir(parents=True)
    pd.DataFrame({"id": ids, "label": labels, "text": ["x"] * len(ids)}).to_csv(
        corpus / "annotation" / "all.csv", index=False
    )
    pd.DataFrame({"id": ids[:30], "annotation_file": "annotation/all.csv"}).to_csv(
        corpus / "list" / "train.csv", index=False
    )
    pd.DataFrame({"id": ids[30:], "annotation_file": "annotation/all.csv"}).to_csv(
        corpus / "list" / "test.csv", index=False
    )

    return run_dir, corpus
