"""
qc_preprocessing.py
====================
QC, doublet detection, normalization for integrated
GSE124395 + GSE189175 human liver scRNA-seq data.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from scipy import stats
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

MIN_GENES=200; MAX_GENES=6000; MAX_MITO=20.0
MIN_COUNTS=500; MAX_COUNTS=50000; MIN_CELLS=3

def compute_qc_metrics(adata):
    logger.info("Computing QC metrics...")
    adata.var["mt"]   = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.match(r"^RP[SL]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt","ribo"],
                               percent_top=[20,50], log1p=True, inplace=True)
    adata.obs["complexity"] = (np.log1p(adata.obs["n_genes_by_counts"])
                               / np.log1p(adata.obs["total_counts"]))
    logger.info(f"  Median genes: {adata.obs['n_genes_by_counts'].median():.0f}")
    logger.info(f"  Median mito%: {adata.obs['pct_counts_mt'].median():.1f}%")
    return adata

def run_scrublet(adata, sample_key="sample_id", threshold=0.25):
    logger.info("Running Scrublet doublet detection...")
    scores = np.zeros(adata.n_obs)
    calls  = np.zeros(adata.n_obs, dtype=bool)
    groups = adata.obs[sample_key] if sample_key in adata.obs else pd.Series(["all"]*adata.n_obs)
    for sid, grp in adata.obs.groupby(groups):
        idx = np.where(groups == sid)[0]
        X   = adata[idx].X
        if issparse(X): X = X.toarray()
        try:
            scrub = scr.Scrublet(X, expected_doublet_rate=0.06)
            s, c  = scrub.scrub_doublets(verbose=False)
            c     = s >= threshold
        except:
            s = np.zeros(len(idx)); c = np.zeros(len(idx), dtype=bool)
        scores[idx] = s; calls[idx] = c
    adata.obs["doublet_score"]     = scores
    adata.obs["predicted_doublet"] = calls
    logger.info(f"  Doublets: {calls.sum():,} ({calls.mean()*100:.1f}%)")
    return adata

def apply_qc_filters(adata):
    n0 = adata.n_obs
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    keep = (
        (adata.obs["n_genes_by_counts"]  >= MIN_GENES) &
        (adata.obs["n_genes_by_counts"]  <= MAX_GENES) &
        (adata.obs["total_counts"]       >= MIN_COUNTS) &
        (adata.obs["total_counts"]       <= MAX_COUNTS) &
        (adata.obs["pct_counts_mt"]      <= MAX_MITO)
    )
    if "predicted_doublet" in adata.obs:
        keep &= ~adata.obs["predicted_doublet"]
    adata = adata[keep].copy()
    logger.success(f"QC: {adata.n_obs:,}/{n0:,} cells retained ({adata.n_obs/n0*100:.1f}%)")
    return adata

def normalize_log(adata):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.layers["normalized"] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
    logger.success("Normalization complete.")
    return adata

def select_hvg(adata, n_top=3000, batch_key="dataset"):
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top,
        flavor="seurat_v3", layer="counts",
        batch_key=batch_key if batch_key in adata.obs else None,
        subset=False)
    n = adata.var["highly_variable"].sum()
    logger.info(f"  HVGs selected: {n:,}")
    return adata

def plot_qc(adata, save_path="results/figures/qc_violin.pdf"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("QC Metrics — GSE124395 + GSE189175", fontweight="bold")
    metrics = [("n_genes_by_counts","Genes/Cell","#3498db"),
               ("total_counts","UMI Counts","#2ecc71"),
               ("pct_counts_mt","Mito %","#e74c3c")]
    col = "dataset" if "dataset" in adata.obs else None
    for ax, (m, lab, c) in zip(axes, metrics):
        if col:
            sns.violinplot(data=adata.obs, x=col, y=m, ax=ax,
                          inner="quartile", color=c, linewidth=0.8)
        else:
            ax.violinplot(adata.obs[m].values)
        ax.set_title(lab, fontweight="bold"); ax.set_xlabel("")
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"QC plot saved -> {save_path}")

def run_full_preprocessing(adata):
    logger.info("="*50)
    logger.info("Starting QC + Preprocessing Pipeline")
    logger.info("="*50)
    adata = compute_qc_metrics(adata)
    adata = run_scrublet(adata)
    plot_qc(adata)
    adata = apply_qc_filters(adata)
    adata = normalize_log(adata)
    adata = select_hvg(adata)
    out = Path("data/processed/qc_filtered.h5ad")
    adata.write_h5ad(out, compression="gzip")
    logger.success(f"Saved -> {out}")
    return adata

if __name__ == "__main__":
    from src.data_ingestion import load_raw
    adata = load_raw()
    run_full_preprocessing(adata)
