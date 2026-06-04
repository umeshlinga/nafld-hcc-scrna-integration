"""
data_ingestion.py
=================
Downloads and loads two real human liver scRNA-seq datasets:
  - GSE124395: Healthy human liver atlas (Aizarani et al. Nature 2019)
                9 donors, ~10,000 cells
  - GSE189175: Human NAFLD-HCC liver (Genome Medicine 2022)
                NAFLD + HCC conditions, ~50,000 cells
"""
import os, gzip, tarfile, urllib.request, warnings
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger
from scipy.io import mmread
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DATASETS = {
    "GSE124395": {
        "condition": "Healthy",
        "description": "Human liver cell atlas",
        "paper": "Aizarani et al. Nature 2019",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE124395&format=file",
    },
    "GSE189175": {
        "condition": "NAFLD_HCC",
        "description": "Human NAFLD-HCC liver snRNA-seq",
        "paper": "Genome Medicine 2022",
        "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE189175&format=file",
    },
}

CONDITION_COLORS = {
    "Healthy":   "#2ecc71",
    "NAFLD":     "#f39c12",
    "HCC":       "#e74c3c",
}

def setup_directories():
    for d in [RAW_DIR, PROCESSED_DIR,
              Path("results/figures"),
              Path("results/tables"),
              Path("results/models")]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Directories initialized.")

def download_dataset(accession, output_dir=RAW_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / f"{accession}_RAW.tar"
    if tar_path.exists():
        logger.info(f"Already downloaded: {tar_path}")
        return output_dir / accession
    url = DATASETS[accession]["url"]
    logger.info(f"Downloading {accession} from NCBI GEO...")
    logger.info(f"Description: {DATASETS[accession]['description']}")
    urllib.request.urlretrieve(url, tar_path)
    extracted = output_dir / accession
    extracted.mkdir(exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(extracted)
    logger.success(f"Extracted {accession} to {extracted}")
    return extracted

def load_10x_sample(sample_dir, sample_id, condition):
    def _find(pat):
        matches = list(Path(sample_dir).glob(pat))
        if not matches:
            raise FileNotFoundError(f"{pat} not found in {sample_dir}")
        return matches[0]
    def _lines(f):
        opener = gzip.open if str(f).endswith(".gz") else open
        with opener(f, "rt") as fh:
            return [l.strip() for l in fh]
    matrix_f   = _find("*matrix*")
    barcodes_f = _find("*barcodes*")
    try:
        features_f = _find("*features*")
    except:
        features_f = _find("*genes*")
    if str(matrix_f).endswith(".gz"):
        with gzip.open(matrix_f, "rb") as f:
            matrix = mmread(f).T.tocsr()
    else:
        matrix = mmread(matrix_f).T.tocsr()
    barcodes = _lines(barcodes_f)
    feat     = _lines(features_f)
    gene_ids   = [l.split("\t")[0] for l in feat]
    gene_names = [l.split("\t")[1] if "\t" in l else l for l in feat]
    adata = ad.AnnData(
        X=csr_matrix(matrix),
        obs=pd.DataFrame(index=[f"{sample_id}_{bc}" for bc in barcodes]),
        var=pd.DataFrame({"gene_ids": gene_ids}, index=gene_names),
    )
    adata.obs["sample_id"]  = sample_id
    adata.obs["condition"]  = condition
    adata.obs["dataset"]    = sample_id.split("_")[0]
    logger.info(f"  {sample_id}: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

def build_combined_anndata():
    logger.info("Building combined AnnData from both datasets...")
    adatas = []
    for acc, meta in DATASETS.items():
        raw_path = RAW_DIR / acc
        if not raw_path.exists():
            logger.warning(f"{acc} not downloaded. Run download_dataset('{acc}') first.")
            continue
        sample_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
        for sd in sample_dirs:
            try:
                adata = load_10x_sample(sd, f"{acc}_{sd.name}", meta["condition"])
                adatas.append(adata)
            except Exception as e:
                logger.warning(f"  Skipping {sd.name}: {e}")
    if not adatas:
        raise RuntimeError("No samples loaded. Check data/download_instructions.md")
    combined = ad.concat(adatas, join="outer", label="batch")
    combined.obs_names_make_unique()
    logger.success(f"Combined: {combined.n_obs:,} cells x {combined.n_vars:,} genes")
    logger.info(f"Conditions: {combined.obs['condition'].value_counts().to_dict()}")
    return combined

def save_raw(adata, path=None):
    if path is None:
        path = PROCESSED_DIR / "raw_combined.h5ad"
    adata.write_h5ad(path, compression="gzip")
    logger.success(f"Saved -> {path}")

def load_raw(path=None):
    if path is None:
        path = PROCESSED_DIR / "raw_combined.h5ad"
    return sc.read_h5ad(path)

if __name__ == "__main__":
    setup_directories()
    for acc in DATASETS:
        download_dataset(acc)
    adata = build_combined_anndata()
    save_raw(adata)
    logger.success("Data ingestion complete.")
