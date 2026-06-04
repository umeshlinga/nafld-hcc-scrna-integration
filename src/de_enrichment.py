"""
de_enrichment.py
================
Differential expression + pathway enrichment + PPI network analysis.
Comparisons: NAFLD vs Healthy, HCC vs Healthy, HCC vs NAFLD
Across all annotated cell types from integrated GSE124395 + GSE189175.
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
import networkx as nx
from loguru import logger
from scipy.sparse import issparse

warnings.filterwarnings("ignore")
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)

try:
    import gseapy as gp
    GSEAPY = True
except ImportError:
    GSEAPY = False

LOGFC = 0.5; PVAL = 0.05
ENRICHR_LIBS = ["KEGG_2021_Human","GO_Biological_Process_2021",
                "Reactome_2022","MSigDB_Hallmark_2020"]

def wilcoxon_de(adata, cell_type, g1, g2):
    ct_col = "cell_type" if "cell_type" in adata.obs else "leiden"
    cond_col = "condition_harmonized" if "condition_harmonized" in adata.obs else "condition"
    mask = (adata.obs[ct_col].str.replace(" ","_") == cell_type.replace(" ","_")) &            adata.obs[cond_col].isin([g1, g2])
    sub = adata[mask].copy()
    if sub.n_obs < 20: return pd.DataFrame()
    sub.obs["group"] = sub.obs[cond_col]
    sc.tl.rank_genes_groups(sub, "group", groups=[g1],
        reference=g2, method="wilcoxon", n_genes=sub.n_vars, use_raw=False)
    res = sc.get.rank_genes_groups_df(sub, group=g1, pval_cutoff=1.0)
    res.columns = ["gene","log2FoldChange","pval","padj","pct_1","pct_2"]
    res["cell_type"]   = cell_type
    res["comparison"]  = f"{g1}_vs_{g2}"
    res["method"]      = "Wilcoxon"
    res["significant"] = (res["padj"] < PVAL) & (res["log2FoldChange"].abs() > LOGFC)
    return res

def run_all_de(adata):
    logger.info("Running DE analysis (all cell types x comparisons)...")
    cond_col = "condition_harmonized" if "condition_harmonized" in adata.obs else "condition"
    ct_col   = "cell_type" if "cell_type" in adata.obs else "leiden"
    comparisons = [("NAFLD","Healthy"),("HCC","Healthy"),("HCC","NAFLD")]
    results = []
    for ct in adata.obs[ct_col].unique():
        conds = adata.obs.loc[adata.obs[ct_col].str.replace(" ","_")==ct.replace(" ","_"), cond_col].unique()
        for g1, g2 in comparisons:
            if g1 not in conds or g2 not in conds: continue
            res = wilcoxon_de(adata, ct, g1, g2)
            if not res.empty: results.append(res)
    if not results: return pd.DataFrame()
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv("results/tables/de_results_all.csv", index=False)
    logger.success(f"DE complete | Tests: {len(combined):,} | Significant: {combined['significant'].sum():,}")
    return combined

def run_enrichment(de_results):
    if not GSEAPY or de_results.empty: return
    logger.info("Running pathway enrichment (Enrichr)...")
    all_enrich = []
    for (comp, ct), grp in de_results[de_results["significant"]].groupby(["comparison","cell_type"]):
        genes = grp["gene"].dropna().unique().tolist()
        if len(genes) < 5: continue
        for lib in ENRICHR_LIBS:
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets=lib,
                                 organism="Human", outdir=None, cutoff=PVAL)
                df = enr.results
                df["comparison"] = comp; df["cell_type"] = ct; df["library"] = lib
                all_enrich.append(df)
            except Exception as e:
                logger.warning(f"  Enrichr failed {lib}: {e}")
    if all_enrich:
        pd.concat(all_enrich).to_csv("results/tables/enrichment_results_all.csv", index=False)
        logger.success("Enrichment saved.")

def plot_volcano(de_results, comp="HCC_vs_Healthy", ct="Hepatocytes"):
    mask = (de_results["comparison"]==comp) &            (de_results["cell_type"].str.replace(" ","_")==ct.replace(" ","_"))
    df = de_results[mask].dropna(subset=["log2FoldChange","pval"]).copy()
    if df.empty: return
    df["-log10p"] = -np.log10(df["pval"].clip(1e-300))
    df["sig"] = "NS"
    df.loc[(df["log2FoldChange"]> LOGFC)&(df["pval"]<PVAL),"sig"] = "Up"
    df.loc[(df["log2FoldChange"]<-LOGFC)&(df["pval"]<PVAL),"sig"] = "Down"
    fig, ax = plt.subplots(figsize=(8, 7))
    for grp, col in [("Up","#e74c3c"),("Down","#3498db"),("NS","#bdc3c7")]:
        s = df[df["sig"]==grp]
        ax.scatter(s["log2FoldChange"], s["-log10p"], c=col,
                   s=10 if grp!="NS" else 5, alpha=0.7,
                   label=f"{grp} (n={len(s):,})", rasterized=True)
    ax.axvline(LOGFC, color="gray", ls="--", lw=1)
    ax.axvline(-LOGFC, color="gray", ls="--", lw=1)
    ax.axhline(-np.log10(PVAL), color="gray", ls="--", lw=1)
    top = df[df["sig"]!="NS"].nlargest(10,"-log10p")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["log2FoldChange"], row["-log10p"]),
                    fontsize=7, ha="center")
    ax.set_xlabel("log2 Fold Change"); ax.set_ylabel("-log10(p-value)")
    ax.set_title(f"Volcano: {ct} | {comp}", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    save = f"results/figures/volcano_{comp}_{ct}.pdf"
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Volcano saved -> {save}")

def build_ppi_network(de_results, comp="HCC_vs_Healthy", ct="Hepatocytes"):
    import urllib.request, json
    mask = (de_results["comparison"]==comp) &            (de_results["cell_type"].str.replace(" ","_")==ct.replace(" ","_")) &            de_results["significant"]
    genes = de_results[mask].nlargest(50,"log2FoldChange")["gene"].tolist()
    if len(genes) < 5: return
    logger.info(f"Building PPI network: {comp} | {ct} | {len(genes)} genes")
    try:
        url = (f"https://string-db.org/api/json/network?"
               f"identifiers={'%0d'.join(genes)}&species=9606"
               f"&required_score=700&caller_identity=nafld_hcc_pipeline")
        with urllib.request.urlopen(url, timeout=30) as r:
            edges = json.loads(r.read())
        G = nx.Graph()
        G.add_nodes_from(genes)
        for e in edges:
            G.add_edge(e["preferredName_A"], e["preferredName_B"],
                       weight=e["score"])
        G.remove_nodes_from(list(nx.isolates(G)))
        pr = nx.pagerank(G, weight="weight")
        pd.DataFrame({"gene": list(pr.keys()),
                      "pagerank": list(pr.values())}).sort_values(
            "pagerank", ascending=False).to_csv(
            f"results/tables/ppi_nodes_{comp}_{ct}.csv", index=False)
        logger.success(f"PPI: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        logger.warning(f"PPI failed: {e}")

def run_full_de(adata):
    logger.info("="*50)
    logger.info("Starting DE + Enrichment + PPI Pipeline")
    logger.info("="*50)
    de = run_all_de(adata)
    if not de.empty:
        for comp in ["HCC_vs_Healthy","NAFLD_vs_Healthy","HCC_vs_NAFLD"]:
            for ct in ["Hepatocytes","Kupffer_cells","Stellate_cells"]:
                plot_volcano(de, comp, ct)
        run_enrichment(de)
        build_ppi_network(de, "HCC_vs_Healthy", "Hepatocytes")
    return de

if __name__ == "__main__":
    adata = sc.read_h5ad("data/processed/clustered_annotated.h5ad")
    run_full_de(adata)
