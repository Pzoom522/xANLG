# xANLG

Data and code for [Understanding Linearity of Cross-Lingual Word Embedding Mappings](https://openreview.net/forum?id=8HuyXvbvqX) (TMLR 2022)

## Data

Please find the cross-lingual word analogy corpus (*xANLG*) in the `/data` folder.

## Code

- `get_emb.py`: Retrieve vectors corresponding to lexicons of *xANLG* from pre-trained word embeddings, then perform pre-processing steps. We process one language pair per time.
- `LRCos`: Please directly use the [Vecto](https://github.com/vecto-ai/vecto) library.
- `validate_analogy.py`: Perform the parallelogram validation algorithm introduced in §4.1.3.
- `linear_map.py`: Find the linear mapping using Generic Procrustes Analysis.

## About
If you like our project or find it useful, please give us a :star: and cite us
```bibtex
@article{xANLG,
title={Understanding Linearity of Cross-Lingual Word Embedding Mappings},
author={Xutan Peng and Mark Stevenson and Chenghua Lin and Chen Li},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=8HuyXvbvqX}
}
```
