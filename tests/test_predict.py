"""CPU-only tests of phase 3 (predict): load a saved classifier, apply it, evaluate, report."""

import pickle

import pytest
from conftest import run_cli


@pytest.fixture
def saved_model(embeddings_file, tmp_path):
    """A fitted and persisted phase-2 model: the artifact phase 3 consumes."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from etp.utils import align_embeddings, load_embeddings

    path, train, test = embeddings_file
    X, y = align_embeddings(load_embeddings(path, "layer_0"), train, "label")
    clf = LinearDiscriminantAnalysis().fit(X, y)

    model_file = tmp_path / "lda_layer_0.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(clf, f)
    return model_file, path, test


def test_predict_lda(saved_model):
    from etp.cli.predict import load_model, predict_and_score
    from etp.utils import load_embeddings

    model_file, path, test = saved_model

    clf = load_model(model_file)  # phase 3 loads the persisted model
    df_embeddings = load_embeddings(path, "layer_0")
    df, accuracy = predict_and_score(clf, df_embeddings, test, "label")
    assert accuracy > 0.9
    assert list(df.columns) == ["reference", "hypothesis", "a", "b"]
    # the rows stay in split order, keyed by the id they were aligned on
    assert list(df.index) == list(test["id"])


def test_report():
    from etp.cli import predict

    reference = ["a", "b", "a", "b"]
    hypothesis = ["a", "b", "b", "xyz"]  # one error + one out-of-set prediction

    accuracy, half_ci = predict.accuracy_with_ci(reference, hypothesis)
    assert accuracy == 50.0 and half_ci > 0

    assert predict.map_unknown(reference, hypothesis) == ["a", "b", "b", "*"]


def test_the_evaluation_names_the_layer_it_belongs_to(caplog):
    # with --layer all the reports follow one another in the job log, so each
    # has to carry the layer that earned it
    import logging

    from etp.cli import predict

    with caplog.at_level(logging.INFO, logger="etp.cli.predict"):
        predict.log_evaluation(["a", "b"], ["a", "b"], "layer_7")

    assert "layer_7" in caplog.text
    assert "accuracy 100.0" in caplog.text
    assert "Wilson 95% CI half-width" in caplog.text


def test_confusion_matrix(tmp_path):
    from etp.cli import predict

    reference = ["a", "b", "a", "b"]
    hypothesis = ["a", "b", "b", "xyz"]

    # row-normalized percentages, one brace-wrapped row per true label (a, b);
    # the out-of-set 'xyz' goes to the trailing unknown column.
    cm_path = tmp_path / "cm.txt"
    predict.save_confusion_matrix(reference, hypothesis, cm_path)
    assert (
        cm_path.read_text() == "# label order (columns): a,b,*\n{50,50,0},\n{0,50,50}\n"
    )


def test_confusion_matrix_rows_sum_to_exactly_100(tmp_path):
    # largest-remainder rounding: 3 equal counts are 33.33% each and must come
    # out as 34,33,33, not 33,33,33
    from etp.cli import predict

    cm_path = tmp_path / "cm.txt"
    predict.save_confusion_matrix(["a", "a", "a"], ["a", "b", "c"], cm_path)

    rows = [
        line.strip("{},")
        for line in cm_path.read_text().splitlines()
        if line.startswith("{")
    ]
    for row in rows:
        assert sum(int(v) for v in row.split(",")) == 100


def _run(phase, run_dir, corpus, *extra):
    return run_cli(
        phase,
        "--run-dir",
        str(run_dir),
        "--corpus-dir",
        str(corpus),
        "--label-column",
        "label",
        *extra,
    )


def test_predict_cli_end_to_end(pipeline_run):
    """`etp train` then `etp predict`: scores + confusion matrix land in <run-dir>/scores."""
    run_dir, corpus = pipeline_run

    train = _run(
        "train", run_dir, corpus, "--corpus-split", "list/train.csv", "--layer", "0"
    )
    assert train.returncode == 0, train.stderr

    predict = _run(
        "predict", run_dir, corpus, "--corpus-split", "list/test.csv", "--layer", "0"
    )
    assert predict.returncode == 0, predict.stderr

    assert (run_dir / "scores" / "lda_layer_0.csv").exists()
    assert (run_dir / "scores" / "lda_layer_0_cm.txt").exists()

    # the report is logged, not written to a file, and the logs go to stderr so
    # they do not cut through the progress bar rich redraws on stdout
    assert "layer_0: accuracy" in predict.stderr
    assert "Wilson 95% CI half-width" in predict.stderr
    assert "Saved scores to" in predict.stderr

    import pandas as pd

    scores = pd.read_csv(run_dir / "scores" / "lda_layer_0.csv")
    assert list(scores.columns) == ["id", "reference", "hypothesis", "a", "b"]
    assert len(scores) == 10  # the test split
    assert (scores["reference"] == scores["hypothesis"]).all()  # separable by design


def test_predict_without_a_trained_model_says_which_file_is_missing(pipeline_run):
    run_dir, corpus = pipeline_run
    r = _run(
        "predict", run_dir, corpus, "--corpus-split", "list/test.csv", "--layer", "0"
    )
    assert r.returncode != 0
    assert "lda_layer_0.pkl" in r.stderr


def test_predict_all_keeps_one_set_of_scores_per_layer(pipeline_run):
    run_dir, corpus = pipeline_run

    train = _run(
        "train", run_dir, corpus, "--corpus-split", "list/train.csv", "--layer", "all"
    )
    assert train.returncode == 0, train.stderr

    predict = _run(
        "predict", run_dir, corpus, "--corpus-split", "list/test.csv", "--layer", "all"
    )
    assert predict.returncode == 0, predict.stderr

    written = sorted(p.name for p in (run_dir / "scores").iterdir())
    assert written == [
        "lda_layer_0.csv",
        "lda_layer_0_cm.txt",
        "lda_layer_1.csv",
        "lda_layer_1_cm.txt",
    ]

    # and each layer reports under its own name
    for layer in ("layer_0", "layer_1"):
        assert f"{layer}: accuracy" in predict.stderr
