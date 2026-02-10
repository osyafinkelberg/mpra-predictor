# MPRA Predictor Models

### Project Overview

This repository contains the code for the training, inference, and interpretation of Massively Parallel Reporter Assay (MPRA) prediction models. 

This project utilizes a transfer-learning approach. Our source models (originally trained to predict chromatin tracks or MPRA activity in other cell types from DNA sequences) are adapted from the following works:

* **Sei:** Chen, K.M., Wong, A.K., Troyanskaya, O.G. et al. *A sequence-based global map of regulatory activity for deciphering human genetics*. Nat Genet 54, 940–949 (2022). [https://doi.org/10.1038/s41588-022-01102-2](https://doi.org/10.1038/s41588-022-01102-2)
* **Malinois:** Gosai, S.J., Castro, R.I., Fuentes, N. et al. *Machine-guided design of cell-type-targeting cis-regulatory elements*. Nature 634, 1211–1220 (2024). [https://doi.org/10.1038/s41586-024-08070-z](https://doi.org/10.1038/s41586-024-08070-z)

---

### Installation

To set up the environment and install the package locally:

```bash
git clone [https://github.com/osyafinkelberg/mpra-predictor.git](https://github.com/osyafinkelberg/mpra-predictor.git)
cd mpra-predictor

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

### Input Data

The example code in this repository was trained on MPRA data covering ~60,000 200bp tiles derived from viral genomes. The assays were performed in 6 cell lines (GM12878, Jurkat, MRC5, A549, HEK293, K562).

**Data Source:** Tommy H. Taslim, Joseph A. Finkelberg, et al. (2025). Global cis-regulatory landscape of double-stranded DNA viruses. bioRxiv: 10.1101/2025.07.20.665756v1
