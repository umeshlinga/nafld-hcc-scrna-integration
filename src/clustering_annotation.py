"""
clustering_annotation.py
=========================
PCA, Harmony batch correction, UMAP, Leiden clustering,
and cell type annotation for integrated human liver data.
Covers: Healthy (GSE124395) + NAFLD/HCC (GSE189175)
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

CELL_COLORS = {
    "Hepatocytes":   "#E8A838", "Kupffer_cells":  "#C0392B",
    "Stellate_cells":"#8E44AD", "LSEC":           "#2980B9",
    "Cholangiocytes":"#27AE60", "NK_T_cells":     "#E74C3C",
    "B_cells":       "#16A085", "Monocytes":      "#D35400",
    "Tumor_cells":   "#2C3E50", "Unknown":        "#BDC3C7",
}

LIVER_MARKERS = {
    "Hepatocytes":   ["ALB","APOB","CYP3A4","FABP1","TF","HP"],
    "Kupffer_cells": ["CD68","VSIG4","MARCO","TIMD4","C1QA","C1QB"],
    "Stellate_cells":["ACTA2","COL1A1","PDGFRB","VIM","THY1","LRAT"],
    "LSEC":          ["LYVE1","STAB2","FCN2","CLEC4G","OIT3","PECAM1"],
    "Cholangiocytes":["KRT7","KRT19","EPCAM","SOX9","CFTR"],
    "NK_T_cells":    ["NKG7","GNLY","CD3D","CD8A","GZMB","PRF1"],
    "B_cells":       ["CD79A","MS4A1","CD19","IGKC","IGHM"],
    "Monocytes":     ["LYZ","S100A8","S100A9","CD14","VCAN"],
    "Tumor_cells":   ["GPC3","AFP","MKI67","TOP2A","PCNA"],
}

def run_pca(adata, n_comps=50):
    logger.info("Running PCA...")
    sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=True,
              svd_solver="arpack")
    var = adata.uns["pca"]["variance_ratio"]
    logger.info(f"  PC1: {var[0]*100:.1f}%  PC2: {var[1]*100:.1f}%")
    return adata

def run_harmony(adata, batch_key="dataset", n_pcs=30):
    try:
        import harmonypy as hm
        logger.info("Running Harmony batch correction...")
        pca = adata.obsm["X_pca"][:, :n_pcs]
        meta = adata.obs[[batch_key]]
        ho = hm.run_harmony(pca, meta, batch_key,
                            max_iter_harmony=30, random_state=42)
        adata.obsm["X_pca_harmony"] = ho.Z_corr.T
        logger.success("Harmony complete.")
    except ImportError:
        logger.warning("harmonypy unavailable.")
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"][:, :n_pcs]
    return adata

def compute_umap(adata, n_pcs=30):
    rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    logger.info(f"Computing UMAP (rep={rep})...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs,
                    use_rep=rep, random_state=42)
    sc.tl.umap(adata, min_dist=0.3, random_state=42)
    adata.obsm["X_umap_2d"] = adata.obsm["X_umap"].copy()
    sc.tl.umap(adata, min_dist=0.3, n_components=3, random_state=42)
    adata.obsm["X_umap_3d"] = adata.obsm["X_umap"].copy()
    adata.obsm["X_umap"]    = adata.obsm["X_umap_2d"]
    logger.success("UMAP 2D + 3D complete.")
    return adata

def leiden_sweep(adata, resolutions=[0.2,0.4,0.6,0.8,1.0,1.2]):
    logger.info("Leiden resolution sweep...")
    records = []
    for res in resolutions:
        key = f"leiden_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=42)
        n = adata.obs[key].nunique()
        labels = LabelEncoder().fit_transform(adata.obs[key])
        sil = silhouette_score(adata.obsm["X_umap"], labels, sample_size=2000)
        records.append({"resolution": res, "n_clusters": n, "silhouette": round(sil,4)})
        logger.info(f"  res={res}: {n} clusters | silhouette={sil:.4f}")
    df = pd.DataFrame(records)
    df.to_csv("results/tables/leiden_sweep.csv", index=False)
    best_res = df.loc[df["silhouette"].idxmax(), "resolution"]
    adata.obs["leiden"] = adata.obs[f"leiden_{best_res}"]
    adata.uns["leiden_resolution"] = best_res
    logger.success(f"Best resolution: {best_res}")
    return adata, df

def annotate_clusters(adata):
    logger.info("Annotating cell types...")
    for ct, markers in LIVER_MARKERS.items():
        present = [g for g in markers if g in adata.var_names]
        if len(present) >= 2:
            sc.tl.score_genes(adata, present, score_name=f"score_{ct}")
    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    annot = {}
    for cl in adata.obs["leiden"].unique():
        mask = adata.obs["leiden"] == cl
        if score_cols:
            best = adata[mask].obs[score_cols].mean().idxmax()
            annot[cl] = best.replace("score_","")
        else:
            annot[cl] = "Unknown"
    adata.obs["cell_type"] = adata.obs["leiden"].map(annot)
    pd.DataFrame({"cluster": list(annot.keys()),
                  "cell_type": list(annot.values())}).to_csv(
        "results/tables/cluster_annotation.csv", index=False)
    logger.success(f"Annotated {len(annot)} clusters.")
    return adata

def plot_umap(adata, save="results/figures/umap_panel.pdf"):
    umap = adata.obsm["X_umap"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("UMAP: Integrated Human Liver (GSE124395 + GSE189175)",
                 fontweight="bold", fontsize=14)
    obs = adata.obs
    for ax, (col, title, cmap_d) in zip(axes, [
        ("cell_type", "Cell Type", CELL_COLORS),
        ("condition", "Condition",
         {"Healthy":"#2ecc71","NAFLD":"#f39c12","HCC":"#e74c3c","NAFLD_HCC":"#e74c3c"}),
        ("dataset", "Dataset", None),
    ]):
        if col not in obs.columns:
            ax.set_visible(False); continue
        cats = obs[col].unique()
        if cmap_d:
            colors = [cmap_d.get(c, "#BDC3C7") for c in obs[col]]
        else:
            pal = plt.cm.get_cmap("tab10", len(cats))
            cmap_d2 = {c: pal(i) for i, c in enumerate(cats)}
            colors = [cmap_d2[c] for c in obs[col]]
        ax.scatter(umap[:,0], umap[:,1], c=colors, s=2, alpha=0.6,
                   rasterized=True)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"UMAP panel saved -> {save}")

def run_full_clustering(adata):
    logger.info("="*50)
    logger.info("Starting Clustering + Annotation Pipeline")
    logger.info("="*50)
    adata = run_pca(adata)
    adata = run_harmony(adata)
    adata = compute_umap(adata)
    adata, _ = leiden_sweep(adata)
    adata = annotate_clusters(adata)
    plot_umap(adata)
    out = Path("data/processed/clustered_annotated.h5ad")
    adata.write_h5ad(out, compression="gzip")
    logger.success(f"Saved -> {out}")
    return adata

if __name__ == "__main__":
    adata = sc.read_h5ad("data/processed/qc_filtered.h5ad")
    run_full_clustering(adata)
