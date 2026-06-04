"""
dataset_integration.py
=======================
Integrates GSE124395 (Healthy) + GSE189175 (NAFLD/HCC) using Harmony.
Handles batch correction, label harmonization, and combined embedding.
"""
import warnings
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

warnings.filterwarnings("ignore")

try:
    import harmonypy as hm
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning("harmonypy not installed — skipping batch correction.")

CONDITION_MAP = {
    "Healthy":    "Healthy",
    "NAFLD_HCC":  "NAFLD",
    "HCC":        "HCC",
}

CONDITION_COLORS = {
    "Healthy": "#2ecc71",
    "NAFLD":   "#f39c12",
    "HCC":     "#e74c3c",
}

def harmonize_conditions(adata):
    logger.info("Harmonizing condition labels across datasets...")
    def _map(c):
        for k, v in CONDITION_MAP.items():
            if k in str(c): return v
        return c
    adata.obs["condition_harmonized"] = adata.obs["condition"].map(_map)
    counts = adata.obs["condition_harmonized"].value_counts()
    logger.info(f"  Condition counts: {counts.to_dict()}")
    return adata

def run_harmony_integration(adata, batch_key="dataset", n_pcs=30):
    if not HARMONY_AVAILABLE:
        logger.warning("Harmony unavailable — using uncorrected PCA.")
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"][:, :n_pcs]
        return adata
    logger.info(f"Running Harmony batch correction (batch_key={batch_key})...")
    pca_embed = adata.obsm["X_pca"][:, :n_pcs]
    meta      = adata.obs[[batch_key]].copy()
    ho = hm.run_harmony(pca_embed, meta, batch_key,
                        max_iter_harmony=30, random_state=42)
    adata.obsm["X_pca_harmony"] = ho.Z_corr.T
    logger.success("Harmony integration complete.")
    return adata

def compute_integration_metrics(adata):
    from sklearn.metrics import silhouette_score
    if "X_pca_harmony" not in adata.obsm:
        return {}
    labels = pd.Categorical(adata.obs["dataset"]).codes
    sil = silhouette_score(
        adata.obsm["X_pca_harmony"][:, :20],
        labels, sample_size=2000
    )
    metrics = {
        "harmony_silhouette": round(float(sil), 4),
        "n_cells_total":      int(adata.n_obs),
        "n_datasets":         int(adata.obs["dataset"].nunique()),
        "n_conditions":       int(adata.obs.get("condition_harmonized",
                                  adata.obs["condition"]).nunique()),
    }
    logger.info(f"Integration metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    adata = sc.read_h5ad("data/processed/qc_filtered.h5ad")
    adata = harmonize_conditions(adata)
    adata = run_harmony_integration(adata)
    metrics = compute_integration_metrics(adata)
    adata.write_h5ad("data/processed/integrated.h5ad", compression="gzip")
    logger.success("Integration complete.")
