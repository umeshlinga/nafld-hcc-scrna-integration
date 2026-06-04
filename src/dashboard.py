"""
dashboard.py
============
Interactive Plotly/Dash dashboard for NAFLD-to-HCC progression analysis.
Integrates GSE124395 (Healthy) + GSE189175 (NAFLD/HCC).
Tabs: Overview | UMAP Explorer | DE Volcano | Pathway Enrichment | ML Results
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
import dash
from dash import dcc, html, Input, Output, dash_table

warnings.filterwarnings("ignore")

app = dash.Dash(__name__,
    title="NAFLD-HCC scRNA-seq Integration Dashboard",
    suppress_callback_exceptions=True)

COLORS = {
    "Healthy": "#2ecc71", "NAFLD": "#f39c12",
    "HCC": "#e74c3c", "NAFLD_HCC": "#e74c3c",
    "bg": "#0d1117", "surface": "#161b22",
    "surface2": "#21262d", "border": "#30363d",
    "text": "#f0f6fc", "muted": "#8b949e",
    "accent": "#58a6ff",
}

CELL_COLORS = {
    "Hepatocytes":"#E8A838","Kupffer_cells":"#C0392B",
    "Stellate_cells":"#8E44AD","LSEC":"#2980B9",
    "Cholangiocytes":"#27AE60","NK_T_cells":"#E74C3C",
    "B_cells":"#16A085","Monocytes":"#D35400",
    "Tumor_cells":"#2C3E50","Unknown":"#7F8C8D",
}

def load_data():
    data = {}
    paths = {
        "de":         "results/tables/de_results_all.csv",
        "enrichment": "results/tables/enrichment_results_all.csv",
        "ml":         "results/tables/all_model_results.csv",
        "leiden":     "results/tables/leiden_sweep.csv",
    }
    for k, p in paths.items():
        data[k] = pd.read_csv(p) if Path(p).exists() else None
    try:
        import scanpy as sc
        adata_path = "data/processed/clustered_annotated.h5ad"
        if Path(adata_path).exists():
            data["adata"] = sc.read_h5ad(adata_path)
    except Exception:
        data["adata"] = None
    return data

def stat_card(title, value, color="#58a6ff"):
    return html.Div([
        html.P(title, style={"margin":"0","fontSize":"12px",
               "color":COLORS["muted"],"textTransform":"uppercase",
               "letterSpacing":"0.06em"}),
        html.H3(value, style={"margin":"4px 0","fontSize":"26px",
                "fontWeight":"700","color":color}),
    ], style={"background":COLORS["surface2"],"borderRadius":"10px",
              "padding":"18px","borderTop":f"3px solid {color}"})

def tab_overview(data):
    adata = data.get("adata")
    n_cells = f"{adata.n_obs:,}" if adata else "~60,000"
    cond_counts = adata.obs["condition"].value_counts().to_dict() if adata else {}

    fig = go.Figure()
    for cond, col in COLORS.items():
        if cond in ["Healthy","NAFLD","HCC","NAFLD_HCC"]:
            n = cond_counts.get(cond, 0)
            if n > 0:
                fig.add_trace(go.Bar(name=cond, x=[cond], y=[n],
                    marker_color=col, text=[f"{n:,}"], textposition="outside"))
    fig.update_layout(title="Cells per Condition", showlegend=False,
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]), margin=dict(l=40,r=20,t=50,b=40))

    return html.Div([
        html.Div([
            stat_card("Total Cells",    n_cells,   "#58a6ff"),
            stat_card("Datasets",       "2",        "#2ecc71"),
            stat_card("Conditions",     "3",        "#f39c12"),
            stat_card("Disease Stages", "Healthy→NAFLD→HCC", "#e74c3c"),
        ], style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)",
                  "gap":"16px","marginBottom":"24px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig)],
                     style={"flex":"1","background":COLORS["surface"],
                            "borderRadius":"10px","padding":"8px"}),
            html.Div([
                html.H3("Datasets", style={"color":COLORS["text"],"marginTop":"0"}),
                html.P("GSE124395 — Healthy Human Liver Atlas",
                       style={"color":COLORS["accent"],"fontWeight":"500"}),
                html.P("Aizarani et al. Nature 2019 | 9 donors | ~10,000 cells",
                       style={"color":COLORS["muted"],"fontSize":"13px"}),
                html.Hr(style={"borderColor":COLORS["border"]}),
                html.P("GSE189175 — Human NAFLD-HCC Liver",
                       style={"color":COLORS["accent"],"fontWeight":"500"}),
                html.P("Genome Medicine 2022 | NAFLD+HCC | ~50,000 cells",
                       style={"color":COLORS["muted"],"fontSize":"13px"}),
                html.Hr(style={"borderColor":COLORS["border"]}),
                html.P("Pipeline: Scanpy · Harmony · CellTypist · PyDESeq2 · "
                       "GSEApy · XGBoost · SHAP · Plotly/Dash",
                       style={"color":COLORS["muted"],"fontSize":"12px"}),
            ], style={"flex":"1","background":COLORS["surface"],
                      "borderRadius":"10px","padding":"20px",
                      "border":f"1px solid {COLORS['border']}"}),
        ], style={"display":"flex","gap":"16px"}),
    ], style={"padding":"24px"})

def tab_umap(data):
    adata = data.get("adata")
    options = ["cell_type","condition","dataset"]
    gene_opts = []
    if adata is not None:
        gene_opts = sorted(adata.var_names.tolist())[:500]

    return html.Div([
        html.Div([
            html.Label("Color by:", style={"color":COLORS["muted"],"fontSize":"13px"}),
            dcc.Dropdown(id="umap-color",
                options=[{"label":v,"value":v} for v in options],
                value="cell_type", clearable=False, style={"color":"#000"}),
            html.Br(),
            html.Label("Gene overlay:", style={"color":COLORS["muted"],"fontSize":"13px"}),
            dcc.Dropdown(id="umap-gene",
                options=[{"label":g,"value":g} for g in gene_opts],
                placeholder="Select gene...", style={"color":"#000"}),
        ], style={"width":"200px","padding":"16px","background":COLORS["surface"],
                  "borderRadius":"10px","border":f"1px solid {COLORS['border']}"}),
        html.Div([dcc.Graph(id="umap-graph", style={"height":"600px"})],
                 style={"flex":"1"}),
    ], style={"display":"flex","gap":"16px","padding":"24px"})

def tab_de(data):
    de = data.get("de")
    comparisons = de["comparison"].unique().tolist() if de is not None else []
    cell_types  = de["cell_type"].unique().tolist()  if de is not None else []
    return html.Div([
        html.Div([
            html.Label("Comparison:", style={"color":COLORS["muted"],"fontSize":"13px"}),
            dcc.Dropdown(id="de-comp",
                options=[{"label":c,"value":c} for c in comparisons],
                value=comparisons[0] if comparisons else None,
                clearable=False, style={"color":"#000"}),
            html.Br(),
            html.Label("Cell Type:", style={"color":COLORS["muted"],"fontSize":"13px"}),
            dcc.Dropdown(id="de-ct",
                options=[{"label":c,"value":c} for c in cell_types],
                value=cell_types[0] if cell_types else None,
                clearable=False, style={"color":"#000"}),
        ], style={"width":"200px","padding":"16px","background":COLORS["surface"],
                  "borderRadius":"10px","border":f"1px solid {COLORS['border']}"}),
        html.Div([
            dcc.Graph(id="volcano-graph", style={"height":"500px"}),
            html.Div(id="de-table"),
        ], style={"flex":"1"}),
    ], style={"display":"flex","gap":"16px","padding":"24px"})

def tab_ml(data):
    ml = data.get("ml")
    fig = go.Figure()
    if ml is not None:
        for comp in ml["comparison"].unique():
            sub = ml[ml["comparison"]==comp].sort_values("roc_auc", ascending=False)
            fig.add_trace(go.Bar(
                name=comp, x=sub["model"], y=sub["roc_auc"],
                text=[f"{v:.3f}" for v in sub["roc_auc"]],
                textposition="outside",
                error_y=dict(type="data",
                    array=sub.get("roc_auc_std", pd.Series([0]*len(sub))).tolist(),
                    visible=True)))
    fig.update_layout(
        title="Model ROC-AUC (5-fold Nested CV)",
        barmode="group", yaxis=dict(range=[0.5,1.05],title="ROC-AUC"),
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        legend=dict(bgcolor=COLORS["surface2"]))
    return html.Div([
        dcc.Graph(figure=fig),
        html.P("Models: XGBoost · LightGBM · Random Forest · SVM · Logistic Regression",
               style={"color":COLORS["muted"],"fontSize":"13px","padding":"0 24px"}),
    ], style={"padding":"24px"})

def build_layout(data):
    TAB = {"background":COLORS["surface"],"color":COLORS["muted"],
           "border":"none","borderBottom":f"2px solid {COLORS['border']}",
           "padding":"12px 20px","fontSize":"14px"}
    STAB = {**TAB,"color":COLORS["text"],
            "borderBottom":f"2px solid {COLORS['accent']}",
            "background":COLORS["surface2"]}
    return html.Div([
        html.Div([
            html.Div([
                html.H1("NAFLD → HCC Progression",
                    style={"margin":"0","fontSize":"20px","fontWeight":"700",
                           "color":COLORS["text"]}),
                html.P("Multi-cohort scRNA-seq Integration | GSE124395 + GSE189175 | ~60,000 human cells",
                    style={"margin":"2px 0 0","fontSize":"12px","color":COLORS["muted"]}),
            ]),
            html.Span("● Live | MIT License",
                style={"color":"#2ecc71","fontSize":"12px","fontWeight":"600"}),
        ], style={"background":COLORS["surface"],
                  "borderBottom":f"1px solid {COLORS['border']}",
                  "padding":"16px 24px","display":"flex",
                  "justifyContent":"space-between","alignItems":"center"}),
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview",    value="overview",    style=TAB, selected_style=STAB),
            dcc.Tab(label="UMAP",        value="umap",        style=TAB, selected_style=STAB),
            dcc.Tab(label="DE Analysis", value="de",          style=TAB, selected_style=STAB),
            dcc.Tab(label="ML Results",  value="ml",          style=TAB, selected_style=STAB),
        ]),
        html.Div(id="tab-content"),
    ], style={"background":COLORS["bg"],"minHeight":"100vh",
              "fontFamily":"Inter, sans-serif"})

def register_callbacks(data):
    @app.callback(Output("tab-content","children"), Input("tabs","value"))
    def render_tab(tab):
        if tab == "overview": return tab_overview(data)
        if tab == "umap":     return tab_umap(data)
        if tab == "de":       return tab_de(data)
        if tab == "ml":       return tab_ml(data)
        return html.Div()

    @app.callback(
        [Output("volcano-graph","figure"), Output("de-table","children")],
        [Input("de-comp","value"), Input("de-ct","value")])
    def update_volcano(comp, ct):
        de = data.get("de")
        if de is None or comp is None: return go.Figure(), html.Div()
        mask = (de["comparison"]==comp) &                (de["cell_type"].str.replace(" ","_")==(ct or "").replace(" ","_"))
        df = de[mask].dropna(subset=["log2FoldChange","pval"]).copy()
        if df.empty: return go.Figure(), html.Div("No data.")
        df["-log10p"] = -np.log10(df["pval"].clip(1e-300))
        df["sig"] = "NS"
        df.loc[(df["log2FoldChange"]>0.5)&(df["pval"]<0.05),"sig"] = "Up"
        df.loc[(df["log2FoldChange"]<-0.5)&(df["pval"]<0.05),"sig"] = "Down"
        fig = px.scatter(df, x="log2FoldChange", y="-log10p",
            color="sig", color_discrete_map={"Up":"#e74c3c","Down":"#3498db","NS":"#6b7280"},
            hover_data=["gene"], opacity=0.7)
        fig.add_vline(x=0.5,  line_dash="dash", line_color="#6b7280", line_width=1)
        fig.add_vline(x=-0.5, line_dash="dash", line_color="#6b7280", line_width=1)
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#6b7280", line_width=1)
        fig.update_layout(title=f"Volcano: {ct} | {comp}",
            paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]))
        table_df = df[df["sig"]!="NS"][["gene","log2FoldChange","pval","padj","sig"]
                   ].round(4).head(50) if "padj" in df.columns else                    df[df["sig"]!="NS"][["gene","log2FoldChange","pval","sig"]].round(4).head(50)
        table = dash_table.DataTable(
            data=table_df.to_dict("records"),
            columns=[{"name":c,"id":c} for c in table_df.columns],
            sort_action="native", filter_action="native", page_size=10,
            style_table={"overflowX":"auto","marginTop":"16px"},
            style_cell={"backgroundColor":COLORS["surface2"],
                        "color":COLORS["text"],"fontSize":"13px"},
            style_header={"backgroundColor":COLORS["surface"],"fontWeight":"bold",
                          "color":COLORS["text"]})
        return fig, table

    @app.callback(Output("umap-graph","figure"),
                  [Input("umap-color","value"), Input("umap-gene","value")])
    def update_umap(color_by, gene):
        adata = data.get("adata")
        if adata is None: return go.Figure()
        umap = adata.obsm["X_umap"]
        obs  = adata.obs.copy()
        obs["UMAP1"] = umap[:,0]; obs["UMAP2"] = umap[:,1]
        if gene and gene in adata.var_names:
            X = adata.X
            if hasattr(X,"toarray"): X = X.toarray()
            obs["expr"] = X[:, adata.var_names.tolist().index(gene)]
            fig = px.scatter(obs, x="UMAP1", y="UMAP2", color="expr",
                color_continuous_scale="Viridis", opacity=0.6,
                title=f"UMAP — {gene} expression")
        else:
            cmap = CELL_COLORS if color_by=="cell_type" else {
                "Healthy":COLORS["Healthy"],"NAFLD":COLORS["NAFLD"],
                "HCC":COLORS["HCC"],"NAFLD_HCC":COLORS["HCC"]}
            if color_by in obs.columns:
                fig = px.scatter(obs, x="UMAP1", y="UMAP2",
                    color=color_by, color_discrete_map=cmap,
                    opacity=0.6, title=f"UMAP — {color_by}")
            else:
                fig = px.scatter(obs, x="UMAP1", y="UMAP2", opacity=0.6)
        fig.update_traces(marker=dict(size=2))
        fig.update_layout(paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]),
            legend=dict(bgcolor=COLORS["surface2"]),
            margin=dict(l=10,r=10,t=40,b=10))
        return fig

if __name__ == "__main__":
    import sys
    logger.info("Loading dashboard data...")
    data = load_data()
    app.layout = build_layout(data)
    register_callbacks(data)
    logger.success("Dashboard ready at http://localhost:8050")
    app.run(debug="--debug" in sys.argv, host="0.0.0.0", port=8050)
