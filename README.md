# HyperGOOD

This is the official pytorch implementation of the paper "HyperGOOD: Towards Out-of-Distribution in Hypergraphs, in AAAI 2026".

![model](https://github.com/ca1man-2022/HyperGOOD/blob/main/cr_model.png)

Also, we provide an appendix [here](https://github.com/ca1man-2022/HyperGOOD/blob/main/Appendix_HyperGOOD.pdf).


## Abstract

Out-of-distribution (OOD) detection plays a critical role in ensuring the robustness of machine learning models in open-world settings. While extensive efforts have been made in vision, language, and graph domains, the challenge of OOD detection in hypergraph-structured data remains unexplored. In this work, we formalize the problem of hypergraph out-of-distribution (HOOD) detection, which aims to identify nodes or hyperedges whose high-order relational contexts differ significantly from those seen during training. We propose HyperGOOD, a unified energy-based detection framework that integrates multi-scale spectral decomposition with structure-aware uncertainty propagation. By preserving both low- and high-frequency signals and diffusing uncertainty across the hypergraph, HyperGOOD effectively captures subtle and relationally entangled anomalies. Experimental results on nine hypergraph datasets demonstrate the effectiveness of our approach, establishing a new foundation for robust hypergraph learning under distributional shifts.




## Citation
If you find our code and idea useful, please cite our work. Thank you!
```python
@inproceedings{cai2026hypergood,
    title = {HyperGOOD: Towards Out-of-Distribution in Hypergraphs},
    author = {Tingyi Cai and Yunliang Jiang and Ming Li and Changqin Huang and Yujie Fang and Chengling Gao and Zhonglong Zheng},
    booktitle = {AAAI},
    year = {2026}
    }
```
