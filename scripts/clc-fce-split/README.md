# The published split of the clc-fce corpus

Three files, one per split, listing the CLC-FCE scripts the corpus keeps:

| file | rows |
| --- | --- |
| `clc-fce-train.csv` | 602 |
| `clc-fce-validation.csv` | 140 |
| `clc-fce-test.csv` | 140 |

One column, `id`: the sortkey Cambridge gives every script of the CLC-FCE dataset, `TR1117*0100*2000*01` (candidate code, exam code, year, session). The built corpus uses it as its own id, so a script is designated by the same string everywhere, from these files to `embeddings.h5`.

The 882 scripts cover the 14 best-represented first languages, balanced at 63 scripts each, and the split is stratified: 43 scripts per language in train, 10 in validation, 10 in test.

`scripts/prepare_clc_fce_data.py` reads these files, downloads the dataset from Cambridge and writes the corpus under `data/clc-fce`. Publishing the split rather than drawing it makes the corpus reproducible: everyone rebuilds the very corpus our reported results were measured on.

## License

These identifiers are derived from the CLC-FCE dataset (Yannakoudakis et al., 2011), whose copyright is held by the University of Cambridge. They are released for non-commercial research and educational purposes only, under the terms of the CLC-FCE Dataset Licence Agreement shipped inside the dataset archive, and not under the MIT license of the rest of this repository. No text, annotation or metadata of the dataset is redistributed here: rebuilding the corpus requires downloading the archive from Cambridge and accepting its licence.

```bibtex
@inproceedings{yannakoudakis-etal-2011-new,
author = {Yannakoudakis, Helen and Briscoe, Ted and Medlock, Ben},
booktitle = {The 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
title = {{A New Dataset and Method for Automatically Grading ESOL Texts}},
year = {2011}
}
```
