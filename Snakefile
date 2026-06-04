configfile: "configs/config.yaml"

rule all:
    input:
        "data/processed/raw_combined.h5ad",
        "data/processed/qc_filtered.h5ad",
        "data/processed/integrated.h5ad",
        "data/processed/clustered_annotated.h5ad",
        "results/tables/de_results_all.csv",
        "results/tables/enrichment_results_all.csv",
        "results/tables/all_model_results.csv",

rule download_data:
    output: directory("data/raw/GSE124395"), directory("data/raw/GSE189175")
    shell: "python -c \"from src.data_ingestion import setup_directories, download_dataset; setup_directories(); [download_dataset(a) for a in ['GSE124395','GSE189175']]\""

rule build_anndata:
    input:  "data/raw/GSE124395", "data/raw/GSE189175"
    output: "data/processed/raw_combined.h5ad"
    shell:  "python -c \"from src.data_ingestion import build_combined_anndata, save_raw; save_raw(build_combined_anndata())\""

rule qc_preprocessing:
    input:  "data/processed/raw_combined.h5ad"
    output: "data/processed/qc_filtered.h5ad"
    shell:  "python src/qc_preprocessing.py"

rule integration:
    input:  "data/processed/qc_filtered.h5ad"
    output: "data/processed/integrated.h5ad"
    shell:  "python src/dataset_integration.py"

rule clustering:
    input:  "data/processed/integrated.h5ad"
    output: "data/processed/clustered_annotated.h5ad"
    shell:  "python src/clustering_annotation.py"

rule de_analysis:
    input:  "data/processed/clustered_annotated.h5ad"
    output: "results/tables/de_results_all.csv"
    shell:  "python src/de_enrichment.py"

rule enrichment:
    input:  "results/tables/de_results_all.csv"
    output: "results/tables/enrichment_results_all.csv"
    shell:  "python -c \"import scanpy as sc, pandas as pd; from src.de_enrichment import save_all_enrichment; save_all_enrichment(pd.read_csv('results/tables/de_results_all.csv'))\""

rule ml_biomarkers:
    input:  "data/processed/clustered_annotated.h5ad", "results/tables/de_results_all.csv"
    output: "results/tables/all_model_results.csv"
    shell:  "python src/ml_biomarker.py"
