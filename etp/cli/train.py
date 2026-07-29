import logging
import pickle
from pathlib import Path

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from etp.utils import (
    align_embeddings,
    available_layers,
    load_corpus_table,
    load_embeddings,
    resolve_layer,
)

logger = logging.getLogger(__name__)


def run_train(
    corpus_dir: str, run_dir: Path, train_list: str, label: str, layer_name: str
):
    train_table = load_corpus_table(corpus_dir, train_list)

    embeddings = run_dir / "embeddings.h5"
    output_dir = run_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = (
        available_layers(embeddings)
        if layer_name == "all"
        else [resolve_layer(embeddings, layer_name)]
    )
    for layer in layers:
        df_embeddings = load_embeddings(embeddings, layer)

        X_train, y_train = align_embeddings(df_embeddings, train_table, label)
        clf = LinearDiscriminantAnalysis().fit(X_train, y_train)

        model_file = output_dir / f"lda_{layer}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(clf, f)

        logger.info(f"Saved model to {model_file}")
