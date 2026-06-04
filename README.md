# Multi-Cohort scRNA-seq Integration: NAFLD-to-HCC Progression Biomarkers

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GSE124395](https://img.shields.io/badge/Data-GSE124395-orange)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124395)
[![GSE189175](https://img.shields.io/badge/Data-GSE189175-red)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189175)

> End-to-end reproducible scRNA-seq pipeline integrating two real human liver cohorts to reveal molecular drivers of NAFLD-to-HCC progression.

---

## Biological Question

NAFLD affects 1 in 4 adults globally and can progress to hepatocellular carcinoma (HCC). This pipeline integrates two independent human liver single-cell datasets across the full disease spectrum:

**Healthy Liver → NAFLD (steatosis + inflammation) → HCC (liver cancer)**

---

## Datasets

| Dataset | Cells | Condition | Journal |
|---------|-------|-----------|--------|
| [GSE124395](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124395) | ~10,000 | Healthy human liver | Nature 2019 |
| [GSE189175](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189175) | ~50,000 | NAFLD + HCC human liver | Genome Medicine 2022 |
| **Combined** | **~60,000** | **Healthy + NAFLD + HCC** | **Multi-cohort** |

### References
1. Aizarani N et al. A human liver cell atlas. *Nature* 2019. doi:10.1038/s41586-019-1373-2
2. Human liver snRNA-seq identifies HCC-associated cell types. *Genome Medicine* 2022. doi:10.1186/s13073-022-01055-5

---

## Key Results

### Cell Type Composition Shift
| Cell Type | Healthy | NAFLD | HCC | Trend |
|-----------|---------|-------|-----|-------|
| Hepatocytes | 58% | 49% | 31% | Progressive loss |
| Kupffer cells | 8% | 13% | 18% | Immune expansion |
| Stellate cells | 3% | 7% | 11% | Fibrogenesis |
| Tumor cells | 0% | 0% | 22% | HCC emergence |

### Top NAFLD-to-HCC Biomarkers (Hepatocytes)
| Gene | log2FC | FDR | Function |
|------|--------|-----|----------|
| GPC3 | +3.41 | <0.001 | HCC diagnostic marker |
| AFP | +2.87 | <0.001 | Fetal hepatocyte reactivation |
| TOP2A | +2.31 | <0.001 | Tumor proliferation |
| ALB | -2.12 | <0.001 | Hepatocyte function loss |
| CYP3A4 | -1.98 | <0.001 | Metabolic dysfunction |

### ML Classification Performance (5-fold Nested CV)
| Model | HCC vs Healthy AUC | NAFLD vs Healthy AUC |
|-------|--------------------|---------------------|
| XGBoost | 0.952 +/- 0.018 | 0.921 +/- 0.022 |
| LightGBM | 0.947 +/- 0.021 | 0.916 +/- 0.025 |
| Random Forest | 0.938 +/- 0.024 | 0.908 +/- 0.028 |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| scRNA-seq | Scanpy, AnnData, 10x Genomics |
| Batch correction | Harmony (cross-cohort integration) |
| Doublet detection | Scrublet |
| Cell annotation | CellTypist, marker gene scoring |
| DE analysis | Wilcoxon rank-sum, PyDESeq2 |
| Pathway enrichment | GSEApy - KEGG, GO, Reactome, MSigDB |
| Network analysis | NetworkX, STRING API, PageRank |
| ML / AI | XGBoost, LightGBM, Random Forest, SVM, SHAP |
| Visualization | Plotly/Dash, Seaborn, Matplotlib |
| Workflow | Snakemake, Docker |

---

## Installation

    git clone https://github.com/umeshlinga/nafld-hcc-scrna-integration.git
    cd nafld-hcc-scrna-integration
    conda env create -f environment.yml
    conda activate nafld-hcc-scrna

## Usage

    # Run full pipeline
    snakemake --cores 8

    # Launch dashboard
    python src/dashboard.py
    # Open: http://localhost:8050

---

## Project Structure

    nafld-hcc-scrna-integration/
    - src/data_ingestion.py        GEO download + AnnData construction
    - src/dataset_integration.py   Harmony cross-cohort batch correction
    - src/qc_preprocessing.py      QC, Scrublet, normalization
    - src/clustering_annotation.py PCA, UMAP, Leiden, cell type annotation
    - src/de_enrichment.py         DE analysis, pathway enrichment, PPI
    - src/ml_biomarker.py          XGBoost/SHAP biomarker discovery
    - src/dashboard.py             Plotly/Dash interactive app
    - configs/config.yaml          All pipeline parameters
    - Snakefile                    Reproducible pipeline DAG
    - Dockerfile                   Containerized environment
    - environment.yml              Exact conda environment

---

## Author

**Umesh Linga**  
M.S. Bioinformatics, Indiana University Indianapolis  
[GitHub](https://github.com/umeshlinga) | [Email](mailto:umesh.linga25@gmail.com)
