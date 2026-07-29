"""Download the public CLC-FCE dataset and format it as the clc-fce corpus.

The CLC-FCE dataset (Yannakoudakis et al., 2011) contains English exam scripts
written by learners of several first languages. It is released by Cambridge
University for non-commercial research purposes (see the license file inside
the archive) at:

  https://s3-eu-west-1.amazonaws.com/ilexir-website-media/fce-released-dataset.zip

Output (under <output-dir>/clc-fce):
  annotation/clc-fce.csv                            id, language_l1, age, ...
  list/clc-fce-{train,validation,test}.csv          annotation_file, id

Usage:
  python scripts/prepare_clc_fce_data.py
  python scripts/prepare_clc_fce_data.py --output-dir /path/corpora
  python scripts/prepare_clc_fce_data.py --output-dir /path/corpora --zip fce-released-dataset.zip
"""

import argparse
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

FCE_URL = (
    "https://s3-eu-west-1.amazonaws.com/ilexir-website-media/fce-released-dataset.zip"
)

CORPUS_NAME = "clc-fce"
SPLIT_DIRECTORY = Path(__file__).resolve().parent / f"{CORPUS_NAME}-split"
SPLIT_NAMES = ("train", "validation", "test")

LANGUAGE_TO_ISO = {
    "Portuguese": "por",
    "Spanish": "spa",
    "Polish": "pol",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Greek": "ell",
    "Russian": "rus",
    "Chinese": "zho",
    "Turkish": "tur",
    "Catalan": "cat",
    "Korean": "kor",
    "Japanese": "jpn",
    "Thai": "tha",
}


def collect_text(node, side: str) -> str:
    """
    Text of `node` for one version of the script: 'i' keeps what the learner
    wrote, 'c' what the examiner corrected it to.

    The <NS> error annotations take four shapes in the dataset, and they nest,
    an error being annotated inside the incorrect (or the corrected) form of
    another one. Each is resolved for the requested side only:

      <NS><i>waken</i><c>woken</c></NS>   replacement: the requested side
      <NS><c>the</c></NS>                 insertion:   nothing for 'i'
      <NS><i>the</i></NS>                 deletion:    nothing for 'c'
      <NS>determinate</NS>                span flagged, no correction offered:
                                          transparent, its text is kept as is
    """
    parts = []
    if node.text:
        parts.append(node.text)

    for child in node:
        if child.tag == "NS":
            wanted = child.find(side)
            if wanted is not None:
                parts.append(collect_text(wanted, side))
            elif child.find("i") is None and child.find("c") is None:
                parts.append(collect_text(child, side))
        else:
            parts.append(collect_text(child, side))

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def extract_text_versions(elem):
    """Return (original_text, corrected_text) for a coded_answer element."""
    paragraphs = elem.findall("p") or [elem]
    versions = []
    for side in ("i", "c"):
        parts = (collect_text(paragraph, side).strip() for paragraph in paragraphs)
        versions.append("\n".join(part for part in parts if part))
    return versions[0], versions[1]


def parse_xml_file(filepath):
    root = ET.parse(filepath).getroot()
    head = root.find("head")

    answers = []
    for answer in head.find("text"):
        coded_answer = answer.find("coded_answer")
        if coded_answer is None:
            continue
        original_text, corrected_text = extract_text_versions(coded_answer)
        if original_text or corrected_text:
            answers.append((original_text, corrected_text))

    return {
        "id": head.attrib.get("sortkey"),
        "language_l1": head.findtext("candidate/personnel/language"),
        "age": head.findtext("candidate/personnel/age"),
        "candidate_score": head.findtext("candidate/score"),
        "text": "\n".join(original for original, _ in answers),
        "corrected_text": "\n".join(corrected for _, corrected in answers),
    }


def read_splits(split_directory: Path) -> dict:
    """The published split: the ids of each split, read from one csv per split."""
    splits = {}
    for name in SPLIT_NAMES:
        path = split_directory / f"{CORPUS_NAME}-{name}.csv"
        table = pd.read_csv(path)
        if "id" not in table.columns:
            raise ValueError(f"The split file {path} should contain an 'id' column")
        splits[name] = list(table["id"])

    ids = [i for split in splits.values() for i in split]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"The splits in {split_directory} are not disjoint, "
            f"{len(ids) - len(set(ids))} ids appear more than once"
        )
    return splits


def build_dataframe(dataset_dir: Path, ids) -> pd.DataFrame:
    """The annotation table of the scripts `ids`, indexed by id and in id order."""
    xml_paths = sorted(dataset_dir.glob("*/*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No */*.xml found under {dataset_dir}")

    scripts = {}
    for path in xml_paths:
        row = parse_xml_file(path)
        scripts[row.pop("id")] = row

    missing = [i for i in ids if i not in scripts]
    if missing:
        raise KeyError(
            f"{len(missing)} of the {len(ids)} ids of the split are absent from "
            f"{dataset_dir}, e.g. {missing[:3]}"
        )

    df = pd.DataFrame([scripts[i] for i in ids], index=pd.Index(ids, name="id"))
    df["language_l1"] = df["language_l1"].map(LANGUAGE_TO_ISO)
    unmapped = df.index[df["language_l1"].isna()]
    if not unmapped.empty:
        raise KeyError(
            f"{len(unmapped)} selected scripts have a first language outside "
            f"LANGUAGE_TO_ISO, e.g. {list(unmapped[:3])}"
        )
    return df


def write_list(ids, path: Path) -> None:
    table = pd.DataFrame(
        {"annotation_file": f"annotation/{CORPUS_NAME}.csv", "id": ids}
    )
    table.to_csv(path, index=False)


def prepare(zip_path: Path, output_dir: Path, split_directory: Path) -> None:
    corpus_dir = output_dir / CORPUS_NAME
    (corpus_dir / "annotation").mkdir(parents=True, exist_ok=True)
    (corpus_dir / "list").mkdir(parents=True, exist_ok=True)

    splits = read_splits(split_directory)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        # scripts live in fce-released-dataset/dataset/<session>/doc*.xml
        # (the separate outliers/ directory is left out)
        df = build_dataframe(
            Path(tmp) / "fce-released-dataset" / "dataset",
            sorted(i for split in splits.values() for i in split),
        )

    df.to_csv(corpus_dir / "annotation" / f"{CORPUS_NAME}.csv")

    for name, ids in splits.items():
        write_list(ids, corpus_dir / "list" / f"{CORPUS_NAME}-{name}.csv")

    counts = df["language_l1"].value_counts()
    print(f"Corpus written to {corpus_dir}")
    print(
        f"{len(df)} scripts, {counts.size} first languages, "
        f"{counts.min()} to {counts.max()} per language"
    )
    print("split: " + ", ".join(f"{name}={len(ids)}" for name, ids in splits.items()))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        type=Path,
        help=f"corpus root (the corpus is written to <output-dir>/{CORPUS_NAME})",
    )
    parser.add_argument(
        "--zip",
        default=None,
        type=Path,
        help="local copy of fce-released-dataset.zip (downloaded if omitted)",
    )
    parser.add_argument(
        "--split-dir",
        default=SPLIT_DIRECTORY,
        type=Path,
        help=f"directory holding the published {CORPUS_NAME}-<split>.csv files",
    )
    args = parser.parse_args()

    if args.zip is not None:
        prepare(args.zip, args.output_dir, args.split_dir)
    else:
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            print(f"Downloading {FCE_URL}")
            urllib.request.urlretrieve(FCE_URL, tmp.name)
            prepare(Path(tmp.name), args.output_dir, args.split_dir)


if __name__ == "__main__":
    main()
