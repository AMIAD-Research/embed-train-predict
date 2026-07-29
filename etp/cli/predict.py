import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix

from etp.utils import (
    align_embeddings,
    available_layers,
    load_corpus_table,
    load_embeddings,
    resolve_layer,
)

logger = logging.getLogger(__name__)


def map_unknown(
    reference: Sequence, hypothesis: Sequence, unknown_label: str = "*"
) -> list:
    """Replace hypotheses outside the reference label set by `unknown_label`."""
    known = set(reference)
    return [h if h in known else unknown_label for h in hypothesis]


def accuracy_with_ci(reference: Sequence, hypothesis: Sequence) -> tuple:
    """Accuracy and Wilson 95% confidence interval half-width, both in percent."""
    correct = sum(r == h for r, h in zip(reference, hypothesis))
    result = binomtest(k=correct, n=len(reference))
    low, high = result.proportion_ci(method="wilson")
    accuracy = round(result.statistic * 100, 1)
    half_ci = round((high - low) / 2 * 100, 1)
    return accuracy, half_ci


def classification_report_df(reference: Sequence, hypothesis: Sequence) -> pd.DataFrame:
    """Per-class precision/recall/f1 in percent (sklearn classification_report)."""
    report = classification_report(
        list(reference), list(hypothesis), output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report)
    return df.T.drop("support", axis=1).transform(lambda x: x * 100).round(1)


def _rows_to_percent(cm: pd.DataFrame) -> list:
    """
    Row-normalize counts to integer percentages summing to exactly 100 per row
    (largest-remainder rounding). Returns a list of int lists.
    """
    result = []
    for counts in cm.to_numpy():
        total = int(counts.sum())
        if total == 0:
            result.append([0] * len(counts))
            continue
        exact = counts / total * 100
        floors = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - floors))[: 100 - int(floors.sum())]:
            floors[i] += 1
        result.append(floors.tolist())
    return result


def save_confusion_matrix(
    reference: Sequence, hypothesis: Sequence, path, unknown_label: str = "*"
) -> None:
    """
    Save the confusion matrix as row-normalized percentages, one true-label row
    per line in brace form. Each row sums to exactly 100. Predictions outside
    the reference label set are grouped under `unknown_label`.
    """
    reference = list(reference)
    hypothesis = map_unknown(reference, hypothesis, unknown_label)
    has_unknown = unknown_label in hypothesis

    labels = sorted(set(reference)) + ([unknown_label] if has_unknown else [])

    matrix = confusion_matrix(reference, hypothesis, labels=labels)
    cm = pd.DataFrame(matrix, index=labels, columns=labels)
    if has_unknown:
        cm = cm.drop(index=unknown_label)

    header = "# label order (columns): " + ",".join(str(label) for label in labels)
    body = ",\n".join(
        "{" + ",".join(str(v) for v in row) + "}" for row in _rows_to_percent(cm)
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + body + "\n")


def log_evaluation(reference: Sequence, hypothesis: Sequence, layer: str) -> None:
    """Log accuracy with Wilson CI and the per-class classification report."""
    accuracy, half_ci = accuracy_with_ci(reference, hypothesis)
    mapped = map_unknown(reference, hypothesis)
    logger.info(
        f"{layer}: accuracy {accuracy} +/- {half_ci} (Wilson 95% CI half-width)\n"
        f"{classification_report_df(reference, mapped)}"
    )


def load_model(path):
    """Load a classifier pickled by the train phase (etp.cli.train.run_train)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_and_score(
    clf, df_embeddings: pd.DataFrame, test_table: pd.DataFrame, label_column: str
) -> tuple[pd.DataFrame, float]:
    """Predict the test-split labels with `clf` and score them against the references."""
    X_test, y_test = align_embeddings(df_embeddings, test_table, label_column)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    df = pd.DataFrame(
        {"reference": y_test, "hypothesis": list(pred)}, index=list(test_table["id"])
    )
    df = pd.concat(
        [df, pd.DataFrame(proba, columns=list(clf.classes_), index=df.index)], axis=1
    )
    accuracy = float((df["reference"] == df["hypothesis"]).mean())
    return df, accuracy


def score_layer(clf, embeddings_file, layer, test_table, label_column, output_dir):
    df_embeddings = load_embeddings(embeddings_file, layer)
    df, accuracy = predict_and_score(clf, df_embeddings, test_table, label_column)

    out_file = output_dir / f"lda_{layer}.csv"
    df.to_csv(out_file, index_label="id")
    logger.info(f"Saved scores to {out_file}")

    cm_file = output_dir / f"lda_{layer}_cm.txt"
    save_confusion_matrix(df["reference"], df["hypothesis"], cm_file)
    logger.info(f"Saved confusion matrix to {cm_file}")

    return df, accuracy


def run_predict(
    corpus_dir: str, run_dir: Path, test_list: str, label: str, layer_name: str
):
    test_table = load_corpus_table(corpus_dir, test_list)

    embeddings = run_dir / "embeddings.h5"
    model_dir = run_dir / "models"
    output_dir = run_dir / "scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = (
        available_layers(embeddings)
        if layer_name == "all"
        else [resolve_layer(embeddings, layer_name)]
    )

    for layer in layers:
        clf = load_model(model_dir / f"lda_{layer}.pkl")
        df, _ = score_layer(clf, embeddings, layer, test_table, label, output_dir)
        log_evaluation(df["reference"], df["hypothesis"], layer)
