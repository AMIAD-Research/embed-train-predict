from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_corpus_table(
    directory: str,
    list_filename: str,
    constants: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a corpus as a single pandas DataFrame."""
    list_path = Path(directory) / list_filename
    table_list = pd.read_csv(list_path, dtype=str)
    if not {"annotation_file", "id"}.issubset(table_list.columns):
        raise ValueError(
            f"The list file {list_path} should contain at least 'annotation_file' and 'id' columns"
        )

    parts = []
    for annotation_file, subset in table_list.groupby("annotation_file"):
        df = pd.read_csv(Path(directory) / annotation_file, dtype=str)
        if "id" not in df.columns:
            raise ValueError(
                f"The annotation file {annotation_file} should contain at least the 'id' column"
            )
        if parts and set(df.columns) != set(parts[0].columns):
            raise ValueError(
                f"The annotation file {annotation_file} should contain the same columns as the previous ones"
            )
        rows = df[df["id"].isin(subset["id"])]

        missing = sorted(set(subset["id"]) - set(rows["id"]))
        if missing:
            raise ValueError(
                f"{len(missing)} of the {len(subset)} ids the list file {list_path} names "
                f"are absent from {annotation_file}, e.g. {missing[:3]}"
            )
        parts.append(rows)
    table = pd.concat(parts, ignore_index=True)

    for key, value in (constants or {}).items():
        table[key] = value
    return table


def concat_corpora(corpus_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Union of several corpora: concatenate their annotation tables."""
    return pd.concat(list(corpus_list), ignore_index=True)


def available_layers(embeddings_file: str, group_name: str = "embeddings") -> list[str]:
    """List the embedding names stored in the file (e.g. layer_0 ... layer_32)."""
    with h5py.File(embeddings_file, "r") as f:
        names = list(f[group_name].keys())
    return sorted(
        names, key=lambda n: tuple(int(t) if t.isdigit() else t for t in n.split("_"))
    )


def resolve_layer(
    embeddings_file: str, layer: str, group_name: str = "embeddings"
) -> str:
    """Resolve a --layer value to an embedding name stored in the file."""
    names = available_layers(embeddings_file, group_name)
    if not names:
        raise ValueError(
            f"No embedding stored in {embeddings_file} under '{group_name}'"
        )

    if layer == "last":
        return names[-1]
    if not layer.isdigit():
        raise ValueError(
            f"--layer must be a layer index, 'last' or 'all', got '{layer}'"
        )

    name = f"layer_{layer}"
    if name not in names:
        raise ValueError(
            f"No '{name}' in {embeddings_file}, available: {', '.join(names)}"
        )
    return name


def load_embeddings(
    embeddings_file: str,
    layer: str,
    group_name: str = "embeddings",
    ids_name: str = "ids",
) -> pd.DataFrame:
    """Load one embedding layer as a DataFrame of shape (N, D) indexed by sample id."""
    with h5py.File(embeddings_file, "r") as f:
        ids = f[ids_name][:].astype(str)
        data = f[f"{group_name}/{layer}"][:]
    return pd.DataFrame(np.asarray(data, dtype=np.float32), index=ids)


def align_embeddings(
    df_embeddings: pd.DataFrame, table: pd.DataFrame, label_column: str
) -> tuple[pd.DataFrame, list]:
    """Align stored embeddings to the labels of one split, matching on 'id'."""
    if label_column not in table.columns:
        raise KeyError(
            f"No '{label_column}' column in the corpus table, available: {', '.join(table.columns)}"
        )

    missing = [i for i in table["id"] if i not in df_embeddings.index]
    if missing:
        raise KeyError(
            f"{len(missing)} of the {len(table)} split ids have no embedding, e.g. {missing[:3]}"
        )

    return df_embeddings.loc[table["id"]], list(table[label_column])
