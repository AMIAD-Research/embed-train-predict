"""CPU-only tests of phase 2 (train): fit a classifier on stored embeddings and save it."""

import pickle

from conftest import run_cli


def test_train_fits_one_lda_per_layer(embeddings_file, tmp_path):
    from etp.cli.train import run_train
    from etp.utils import available_layers, load_embeddings

    path, train, _ = embeddings_file
    assert available_layers(path) == ["layer_0"]

    df_embeddings = load_embeddings(path, "layer_0")
    assert df_embeddings.shape == (40, 4) and df_embeddings.index[0] == "s00"

    # run_train reads the split from the corpus, so lay one out next to the h5
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "embeddings.h5").symlink_to(path)

    corpus = tmp_path / "corpus"
    (corpus / "list").mkdir(parents=True)
    (corpus / "annotation").mkdir(parents=True)
    train.to_csv(corpus / "annotation" / "train.csv", index=False)
    train.assign(annotation_file="annotation/train.csv")[
        ["id", "annotation_file"]
    ].to_csv(corpus / "list" / "train.csv", index=False)

    run_train(corpus, run_dir, "list/train.csv", "label", "0")

    model_file = run_dir / "models" / "lda_layer_0.pkl"
    assert model_file.exists()
    with open(model_file, "rb") as f:
        clf = pickle.load(f)
    assert list(clf.classes_) == ["a", "b"]  # fitted on both labels


def test_train_cli_end_to_end(pipeline_run):
    """`etp train` wires --run-dir + corpus into <run-dir>/models/lda_<layer>.pkl."""
    run_dir, corpus = pipeline_run
    r = run_cli(
        "train",
        "--run-dir",
        str(run_dir),
        "--corpus-dir",
        str(corpus),
        "--corpus-split",
        "list/train.csv",
        "--label-column",
        "label",
        "--layer",
        "0",
    )
    assert r.returncode == 0, r.stderr
    assert (run_dir / "models" / "lda_layer_0.pkl").exists()
    # only the layer that was asked for
    assert not (run_dir / "models" / "lda_layer_1.pkl").exists()


def test_train_all_fits_every_stored_layer(pipeline_run):
    run_dir, corpus = pipeline_run
    r = run_cli(
        "train",
        "--run-dir",
        str(run_dir),
        "--corpus-dir",
        str(corpus),
        "--corpus-split",
        "list/train.csv",
        "--label-column",
        "label",
        "--layer",
        "all",
    )
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (run_dir / "models").iterdir()) == [
        "lda_layer_0.pkl",
        "lda_layer_1.pkl",
    ]


def test_train_last_picks_the_deepest_layer(pipeline_run):
    # 'last' is the default, and the fixture stores layer_0 and layer_1
    run_dir, corpus = pipeline_run
    r = run_cli(
        "train",
        "--run-dir",
        str(run_dir),
        "--corpus-dir",
        str(corpus),
        "--corpus-split",
        "list/train.csv",
        "--label-column",
        "label",
    )
    assert r.returncode == 0, r.stderr
    assert [p.name for p in (run_dir / "models").iterdir()] == ["lda_layer_1.pkl"]


def test_an_unknown_layer_is_reported_with_what_the_file_holds(pipeline_run):
    run_dir, corpus = pipeline_run
    r = run_cli(
        "train",
        "--run-dir",
        str(run_dir),
        "--corpus-dir",
        str(corpus),
        "--corpus-split",
        "list/train.csv",
        "--label-column",
        "label",
        "--layer",
        "99",
    )
    assert r.returncode != 0
    assert "layer_0, layer_1" in r.stderr


def test_an_unknown_label_column_is_reported_with_what_the_corpus_holds(pipeline_run):
    run_dir, corpus = pipeline_run
    r = run_cli(
        "train",
        "--run-dir",
        str(run_dir),
        "--corpus-dir",
        str(corpus),
        "--corpus-split",
        "list/train.csv",
        "--label-column",
        "nope",
        "--layer",
        "0",
    )
    assert r.returncode != 0
    assert "nope" in r.stderr and "label" in r.stderr
