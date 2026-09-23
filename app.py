"""Connect Cloud Shiny viewer for the web-slim CSF-PBMC atlas."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import sparse
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
CATEGORIES = json.loads(
    (DATA / "metadata_categories.json").read_text(encoding="utf-8")
)
ARRAYS = np.load(DATA / "atlas_metadata.npz")
MARKERS = pd.read_csv(DATA / "cell_type_top50_marker_catalog.csv")

obs = pd.DataFrame(
    {
        field: pd.Categorical.from_codes(
            ARRAYS[f"{field}_codes"], categories=levels
        )
        for field, levels in CATEGORIES.items()
    }
)
for field in MANIFEST.get("numeric_obs_columns", []):
    obs[field] = np.asarray(ARRAYS[f"{field}_values"], dtype=np.float32)
xy = np.asarray(ARRAYS["X_umap"], dtype=np.float32)
sample_uid = obs["GSE"].astype(str) + "|" + obs["GSM"].astype(str)
expression_scale = float(MANIFEST["expression_scale"])
gene_lookup = MANIFEST["gene_lookup"]
genes = [
    gene
    for part in MANIFEST["expression_parts"]
    for gene in part["genes"]
]
MAX_RENDERED_CELLS = 80_000
MAX_VIOLIN_CELLS_PER_GROUP = 5_000
QC_FIELDS = [
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
    "pct_counts_rb",
]
SUBTYPE_ORDER = [
    "CD4 Naive", "CD8 Naive", "CD4 Memory", "CD8 Memory", "Treg",
    "CD4 CTL", "CD8 CTL", "MAIT", "γδT", "NKT",
    "cMonocytes", "ncMonocytes", "mdMac", "trMac", "infMac", "cDC2",
    "ASDC", "pDC", "Naive B", "Memory B", "Plasma cell", "NK bright",
    "Transit. NK", "NK dim", "ILC2",
]
CELL_TYPE_ORDER = [
    "T cell", "Myeloid cell", "pDC", "B cell", "Innate lymphoid cell"
]

SUBTYPE_COLORS = dict(
    zip(
        [
            "Naive B", "Memory B", "Plasma cell", "CD4 Naive", "CD8 Naive",
            "CD4 Memory", "CD8 Memory", "Treg", "CD4 CTL", "CD8 CTL",
            "MAIT", "γδT", "NKT", "NK bright", "Transit. NK", "NK dim",
            "ILC2", "cMonocytes", "ncMonocytes", "mdMac", "trMac", "infMac",
            "cDC2", "pDC", "ASDC",
        ],
        [
            "#C96F6F", "#B85C68", "#B77AB5", "#7DA7CF", "#4F9A78",
            "#AEC9E4", "#98C970", "#3F6BA7", "#3E7C9D", "#C3D894",
            "#75B38A", "#A6B85A", "#BFB552", "#D7BE66", "#A995C1",
            "#C99235", "#DDD36A", "#C97891", "#9E5F73", "#9568B1",
            "#D6A196", "#C17DB4", "#B46AB0", "#7F8CC1", "#B79BD0",
        ],
    )
)
DISEASE_COLORS = {
    "Younger control": "#1f77b4", "Older control": "#aec7e8",
    "Unknown": "#ff7f0e", "MCI/AD": "#ffbb78", "PD": "#2ca02c",
    "IIH": "#98df8a", "MS": "#d62728", "Ab-mediated IDD": "#ff9896",
    "VE": "#9467bd", "Neuro-COVID": "#8c564b", "CAR-T-DIPG": "#c49c94",
    "NT1": "#e377c2", "CIS": "#f7b6d2", "RIS": "#7f7f7f",
    "DLB": "#c7c7c7", "PHN": "#bcbd22", "Uveitis": "#dbdb8d",
    "NMO": "#17becf", "NT2": "#9edae5",
}
PALETTES = {
    "cell_subtype_short": SUBTYPE_COLORS,
    "cell_type": {
        "B cell": "#B85C68", "T cell": "#4F86B6",
        "Innate lymphoid cell": "#C99235", "Myeloid cell": "#A95C68",
        "pDC": "#7F8CC1",
    },
    "Disease": DISEASE_COLORS,
    "tissue": {"CSF": "#596FA8", "PBMC": "#4F8751"},
    "gender": {"Female": "#D46A92", "Male": "#4C78A8", "Unknown": "#8A8A8A"},
}
fallback = px.colors.qualitative.Safe + px.colors.qualitative.Set3
for field in ["cell_subtype_short", "cell_type", "Disease", "Disease1", "tissue", "GSE", "GSM", "gender"]:
    palette = PALETTES.setdefault(field, {})
    for i, value in enumerate(CATEGORIES[field]):
        palette.setdefault(value, fallback[i % len(fallback)])


@lru_cache(maxsize=2)
def expression_part(part: int) -> sparse.csc_matrix:
    return sparse.load_npz(DATA / f"expression_part_{part:02d}.npz").tocsc()


@lru_cache(maxsize=12)
def expression_vector(gene: str) -> np.ndarray:
    location = gene_lookup[gene.upper()]
    values = expression_part(location["part"])[:, location["column"]]
    return np.asarray(values.toarray(), dtype=np.float32).ravel() / expression_scale


def choices(field: str) -> list[str]:
    return list(CATEGORIES[field])


def layout(title: str, legend: str = "") -> dict:
    return {
        "title": {"text": title, "x": 0, "xanchor": "left", "font": {"size": 18}},
        "paper_bgcolor": "white", "plot_bgcolor": "white",
        "font": {"family": "Arial, Helvetica, sans-serif", "color": "#22262A"},
        "margin": {"l": 48, "r": 20, "t": 58, "b": 48},
        "legend": {
            "title": {"text": legend}, "x": 1.01, "y": 1,
            "groupclick": "togglegroup", "itemclick": "toggle",
            "itemdoubleclick": "toggleothers", "traceorder": "normal",
        },
    }


def category_order(field: str) -> list[str]:
    if field == "cell_subtype_short":
        return SUBTYPE_ORDER
    if field == "cell_type":
        return CELL_TYPE_ORDER
    return choices(field)


def add_large_clickable_legend(
    figure: go.Figure,
    field: str,
    categories: list[str],
) -> None:
    """Use separate legend traces so symbols can grow without growing cells."""
    real_traces = list(figure.data)
    present = {str(trace.name) for trace in real_traces}
    for trace in real_traces:
        trace.legendgroup = str(trace.name)
        trace.showlegend = False
    for category in categories:
        if category not in present:
            continue
        figure.add_trace(
            go.Scattergl(
                x=[None], y=[None], mode="markers", name=category,
                legendgroup=category, showlegend=True, hoverinfo="skip",
                marker={"size": 8.8, "color": PALETTES[field][category]},
            )
        )


def stratified_indices(indices: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Deterministically sample at most N cells per displayed group."""
    rng = np.random.default_rng(2026)
    selected = []
    values = groups.astype(str).to_numpy()
    for value in pd.unique(values):
        group_indices = indices[values == value]
        if len(group_indices) > MAX_VIOLIN_CELLS_PER_GROUP:
            group_indices = rng.choice(
                group_indices, MAX_VIOLIN_CELLS_PER_GROUP, replace=False
            )
        selected.append(np.sort(group_indices))
    return np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)


def box_statistics(values: np.ndarray) -> tuple[float, float, float, float, float]:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower_candidates = values[values >= q1 - 1.5 * iqr]
    upper_candidates = values[values <= q3 + 1.5 * iqr]
    lower = float(lower_candidates.min())
    upper = float(upper_candidates.max())
    return float(q1), float(median), float(q3), lower, upper


def composition_summary(indices: np.ndarray, group: str) -> pd.DataFrame:
    """Return per-group normalized medians of sample-level subtype fractions."""
    frame_columns = list(
        dict.fromkeys(["GSE", "GSM", group, "cell_subtype_short"])
    )
    frame = obs.iloc[indices][frame_columns].copy()
    counts = (
        frame.groupby(frame_columns, observed=True)
        .size()
        .rename("Cells")
        .reset_index()
    )
    totals = counts.groupby(["GSE", "GSM"], observed=True)["Cells"].transform(
        "sum"
    )
    counts["Proportion"] = counts["Cells"] / totals
    summary = (
        counts.groupby([group, "cell_subtype_short"], observed=True)["Proportion"]
        .median()
        .reset_index()
    )
    median_totals = summary.groupby(group, observed=True)["Proportion"].transform(
        "sum"
    )
    summary["Proportion"] = np.divide(
        summary["Proportion"],
        median_totals,
        out=np.zeros(len(summary), dtype=np.float64),
        where=median_totals.to_numpy() > 0,
    )
    return summary


app_ui = ui.page_fillable(
    ui.include_css(ROOT / "www/style.css"),
    ui.tags.header(
        ui.tags.div(
            ui.tags.h1("Human CSF-PBMC Single-Cell Atlas"),
            ui.tags.p("Interactive view of the curated publication cohort"),
            class_="brand",
        ),
        ui.tags.div(ui.tags.span("573,569 curated cells", class_="status-label"), class_="header-status"),
        class_="app-header",
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.tags.h2("Filters"),
            ui.input_selectize("tissue", "Tissue", choices("tissue"), multiple=True),
            ui.input_selectize("disease", "Disease", choices("Disease"), multiple=True),
            ui.input_selectize("disease1", "Disease1", choices("Disease1"), multiple=True),
            ui.input_selectize("gse", "GSE", choices("GSE"), multiple=True),
            ui.input_selectize("gender", "Gender", choices("gender"), multiple=True),
            ui.input_selectize("cell_type", "Cell type", choices("cell_type"), multiple=True),
            ui.input_action_button("reset", "Reset filters", class_="btn-outline-secondary reset-button"),
            ui.tags.hr(),
            ui.tags.p("Disease and Disease1 are retained; the redundant lowercase disease field is excluded.", class_="sidebar-note"),
            width=300,
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Atlas",
                ui.layout_columns(
                    ui.value_box("Cells", ui.output_text("cell_count")),
                    ui.value_box("Samples", ui.output_text("sample_count")),
                    ui.value_box("Datasets", ui.output_text("dataset_count")),
                    ui.value_box("Subtypes", ui.output_text("subtype_count")),
                    col_widths=[3, 3, 3, 3],
                ),
                ui.card(
                    ui.card_header(
                        ui.tags.div(
                            ui.input_select("color_by", "Color by", {
                                "cell_type": "Cell type", "cell_subtype_short": "Cell subtype",
                                "Disease": "Disease", "Disease1": "Disease1", "tissue": "Tissue",
                                "GSE": "GSE", "gender": "Gender",
                                "n_genes_by_counts": "Genes detected",
                                "total_counts": "Total counts",
                                "pct_counts_mt": "Mitochondrial counts (%)",
                                "pct_counts_rb": "Ribosomal counts (%)",
                            }, selected="cell_type"),
                            ui.input_slider("point_limit", "Maximum displayed cells", 10_000, MAX_RENDERED_CELLS, 50_000, step=10_000),
                            class_="inline-controls",
                        )
                    ),
                    output_widget("umap_plot"),
                    ui.tags.p(
                        "Click a legend item to show or hide a cell population; double-click to isolate it.",
                        class_="plot-hint",
                    ),
                    full_screen=True, class_="plot-card",
                ),
            ),
            ui.nav_panel(
                "Marker expression",
                ui.card(
                    ui.card_header(ui.input_selectize("gene", "Marker gene", genes, selected="CD3E" if "CD3E" in genes else genes[0])),
                    output_widget("gene_plot"), full_screen=True, class_="plot-card",
                ),
            ),
            ui.nav_panel(
                "Composition",
                ui.card(
                    ui.card_header(ui.input_select("composition_group", "Group by", {"tissue": "Tissue", "Disease": "Disease", "Disease1": "Disease1", "GSE": "GSE"}, selected="tissue")),
                    output_widget("composition_plot"), full_screen=True, class_="plot-card",
                ),
            ),
            ui.nav_panel(
                "Violin / Boxplot",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Plot controls"),
                        ui.input_select(
                            "violin_group", "Group by",
                            {
                                "tissue": "Tissue", "GSE": "GSE", "GSM": "GSM",
                                "Disease": "Disease", "Disease1": "Disease1",
                                "gender": "Gender", "cell_type": "Cell type",
                                "cell_subtype_short": "Cell subtype",
                            }, selected="GSE",
                        ),
                        ui.input_radio_buttons(
                            "violin_value_type", "Y-axis data",
                            {"qc": "QC metric", "gene": "Marker gene"},
                            selected="qc", inline=True,
                        ),
                        ui.panel_conditional(
                            "input.violin_value_type === 'qc'",
                            ui.input_select(
                                "violin_metric", "QC metric",
                                {
                                    "n_genes_by_counts": "Genes detected",
                                    "total_counts": "Total counts",
                                    "pct_counts_mt": "Mitochondrial counts (%)",
                                    "pct_counts_rb": "Ribosomal counts (%)",
                                }, selected="n_genes_by_counts",
                            ),
                        ),
                        ui.panel_conditional(
                            "input.violin_value_type === 'gene'",
                            ui.input_selectize(
                                "violin_gene", "Marker gene", genes,
                                selected="CD3E" if "CD3E" in genes else genes[0],
                            ),
                        ),
                        ui.input_radio_buttons(
                            "violin_plot_type", "Plot type",
                            {
                                "combined": "Violin + boxplot",
                                "violin": "Violin",
                                "box": "Boxplot",
                            }, selected="combined",
                        ),
                        ui.input_checkbox("violin_points", "Show data points", False),
                        ui.input_select(
                            "violin_scale", "Y-axis scale",
                            {"linear": "Linear", "log10": "log10(value + 1)"},
                            selected="linear",
                        ),
                        ui.tags.p(ui.output_text("violin_note"), class_="sidebar-note"),
                    ),
                    ui.card(
                        output_widget("violin_plot"),
                        full_screen=True, class_="plot-card",
                    ),
                    col_widths=[3, 9],
                ),
            ),
            ui.nav_panel(
                "Marker catalog",
                ui.card(
                    ui.card_header(ui.input_select("marker_group", "Cell type", choices("cell_type"), selected=choices("cell_type")[0])),
                    ui.output_data_frame("marker_table"), full_screen=True, class_="table-card",
                ),
            ),
            ui.nav_panel(
                "Samples",
                ui.card(
                    ui.card_header(ui.tags.div(ui.tags.h2("Filtered sample metadata"), ui.download_button("download_samples", "Download CSV", class_="btn-primary"), class_="table-heading")),
                    ui.output_data_frame("sample_table"), full_screen=True, class_="table-card",
                ),
            ),
            ui.nav_panel(
                "About",
                ui.card(
                    ui.tags.h2("About this atlas"),
                    ui.tags.p("This read-only application displays existing annotations and UMAP coordinates. It does not recompute PCA, Harmony, clustering, or UMAP."),
                    ui.tags.dl(
                        ui.tags.dt("Cells"), ui.tags.dd(f"{len(obs):,}"),
                        ui.tags.dt("Expression"), ui.tags.dd("220 unique genes from the Top 50 DEGs of each major cell type; log1p counts per 10,000."),
                        ui.tags.dt("Metadata"), ui.tags.dd("GSM, GSE, tissue, gender, age, Disease, Disease1, cell type, cell subtype, and four cell-level QC metrics."),
                    ), class_="about-card",
                ),
            ),
        ),
    ),
    title="Human CSF-PBMC Single-Cell Atlas",
)


def server(input, output, session):
    @reactive.effect
    @reactive.event(input.reset)
    def reset_filters():
        for name in ["tissue", "disease", "disease1", "gse", "gender", "cell_type"]:
            ui.update_selectize(name, selected=[])

    @reactive.calc
    def filtered_indices() -> np.ndarray:
        mask = np.ones(len(obs), dtype=bool)
        selections = {
            "tissue": input.tissue(), "Disease": input.disease(),
            "Disease1": input.disease1(), "GSE": input.gse(),
            "gender": input.gender(), "cell_type": input.cell_type(),
        }
        for field, selected in selections.items():
            if selected:
                mask &= obs[field].isin(selected).to_numpy()
        return np.flatnonzero(mask)

    @reactive.calc
    def display_indices() -> np.ndarray:
        selected = filtered_indices()
        limit = int(input.point_limit())
        if len(selected) <= limit:
            return selected
        return np.sort(np.random.default_rng(2026).choice(selected, limit, replace=False))

    @output
    @render.text
    def cell_count(): return f"{len(filtered_indices()):,}"

    @output
    @render.text
    def sample_count(): return f"{sample_uid.iloc[filtered_indices()].nunique():,}"

    @output
    @render.text
    def dataset_count(): return f"{obs.iloc[filtered_indices()]['GSE'].nunique():,}"

    @output
    @render.text
    def subtype_count(): return f"{obs.iloc[filtered_indices()]['cell_subtype_short'].nunique():,}"

    @output
    @render_plotly
    def umap_plot():
        indices = display_indices()
        field = input.color_by()
        frame = obs.iloc[indices][["GSM", "GSE", "tissue", "Disease", "Disease1", "gender", "cell_subtype_short"]].copy()
        frame["UMAP1"], frame["UMAP2"] = xy[indices, 0], xy[indices, 1]
        if field in QC_FIELDS:
            frame["Color"] = obs.iloc[indices][field].to_numpy(dtype=np.float32)
            figure = px.scatter(
                frame, x="UMAP1", y="UMAP2", color="Color",
                color_continuous_scale="Viridis",
                hover_data=["GSM", "GSE", "tissue", "Disease", "Disease1", "gender"],
                render_mode="webgl",
            )
            figure.update_coloraxes(colorbar_title=field)
        else:
            frame["Color"] = obs.iloc[indices][field].astype(str).to_numpy()
            order = category_order(field)
            figure = px.scatter(
                frame, x="UMAP1", y="UMAP2", color="Color",
                color_discrete_map=PALETTES[field],
                category_orders={"Color": order},
                hover_data=["GSM", "GSE", "tissue", "Disease", "Disease1", "gender"],
                render_mode="webgl",
            )
        figure.update_traces(marker={"size": 2.2, "opacity": 0.72})
        if field not in QC_FIELDS:
            add_large_clickable_legend(figure, field, category_order(field))
        figure.update_layout(**layout(f"UMAP colored by {field}", field), dragmode="pan")
        figure.update_xaxes(visible=False); figure.update_yaxes(visible=False, scaleanchor="x")
        return figure

    @output
    @render_plotly
    def gene_plot():
        indices = display_indices()
        gene = input.gene()
        frame = pd.DataFrame({"UMAP1": xy[indices, 0], "UMAP2": xy[indices, 1], "Expression": expression_vector(gene)[indices]})
        frame = frame.sort_values("Expression")
        figure = px.scatter(frame, x="UMAP1", y="UMAP2", color="Expression", color_continuous_scale="Viridis", render_mode="webgl")
        figure.update_traces(marker={"size": 2.2, "opacity": 0.78})
        figure.update_layout(**layout(f"Marker gene expression: {gene}"), dragmode="pan")
        figure.update_xaxes(visible=False); figure.update_yaxes(visible=False, scaleanchor="x")
        return figure

    @output
    @render_plotly
    def composition_plot():
        indices = filtered_indices()
        group = input.composition_group()
        summary = composition_summary(indices, group)
        figure = px.bar(
            summary, x=group, y="Proportion", color="cell_subtype_short",
            color_discrete_map=SUBTYPE_COLORS,
            category_orders={"cell_subtype_short": SUBTYPE_ORDER},
        )
        figure.update_layout(**layout(f"Normalized median sample-level composition by {group}", "Cell subtype"), barmode="stack")
        figure.update_yaxes(tickformat=".0%", title="Normalized median proportion", range=[0, 1])
        if group == "tissue":
            present = [value for value in choices("tissue") if value in set(summary[group].astype(str))]
            figure.update_traces(width=0.46)
            figure.update_xaxes(range=[-0.75, max(len(present) - 0.25, 0.75)])
        return figure

    @reactive.calc
    def violin_payload():
        indices = filtered_indices()
        group = input.violin_group()
        if input.violin_value_type() == "gene":
            gene = input.violin_gene()
            values = expression_vector(gene)
            label = f"{gene} expression"
        else:
            metric = input.violin_metric()
            values = obs[metric].to_numpy(dtype=np.float32)
            label = metric
        selected_values = np.asarray(values[indices], dtype=np.float32)
        valid = np.isfinite(selected_values)
        indices = indices[valid]
        selected_values = selected_values[valid]
        if input.violin_scale() == "log10":
            selected_values = np.log10(np.maximum(selected_values, 0) + 1)
            label = f"log10({label} + 1)"
        full_values = np.asarray(values, dtype=np.float32).copy()
        if input.violin_scale() == "log10":
            full_values = np.log10(np.maximum(full_values, 0) + 1)
        sampled = stratified_indices(indices, obs.iloc[indices][group])
        return indices, sampled, group, full_values, label

    @output
    @render.text
    def violin_note():
        indices, sampled, _, _, _ = violin_payload()
        return (
            f"Statistics use all {len(indices):,} filtered cells; "
            f"violin density displays {len(sampled):,} stratified cells."
        )

    @output
    @render_plotly
    def violin_plot():
        indices, sampled, group, values, label = violin_payload()
        figure = go.Figure()
        groups_all = obs.iloc[indices][group].astype(str).to_numpy()
        groups_sampled = obs.iloc[sampled][group].astype(str).to_numpy()
        present = set(groups_all)
        order = [value for value in category_order(group) if value in present]
        plot_type = input.violin_plot_type()
        show_violin = plot_type in {"combined", "violin"}
        show_box = plot_type in {"combined", "box"}

        for value in order:
            color = PALETTES[group][value]
            if show_violin:
                group_sample = sampled[groups_sampled == value]
                figure.add_trace(
                    go.Violin(
                        x=np.repeat(value, len(group_sample)),
                        y=values[group_sample],
                        name=value, legendgroup=value, showlegend=False,
                        line_color=color, fillcolor=color, opacity=0.68,
                        points="all" if input.violin_points() else False,
                        jitter=0.22, pointpos=0, marker={"size": 2, "opacity": 0.45},
                        spanmode="hard", meanline_visible=False,
                    )
                )
            if show_box:
                group_indices = indices[groups_all == value]
                q1, median, q3, lower, upper = box_statistics(values[group_indices])
                figure.add_trace(
                    go.Box(
                        x=[value], q1=[q1], median=[median], q3=[q3],
                        lowerfence=[lower], upperfence=[upper], name=value,
                        legendgroup=value, showlegend=False, width=0.22,
                        line={"color": "#25292C", "width": 1.2},
                        fillcolor="rgba(255,255,255,0.78)", boxpoints=False,
                    )
                )

        figure.update_layout(
            **layout(f"{label} by {group}"),
            violinmode="overlay", boxmode="overlay", hovermode="closest",
        )
        figure.update_xaxes(
            title=group, categoryorder="array", categoryarray=order,
            tickangle=-45 if len(order) > 8 else 0,
        )
        figure.update_yaxes(title=label, rangemode="tozero")
        return figure

    @output
    @render.data_frame
    def marker_table():
        table = MARKERS.loc[MARKERS["group"] == input.marker_group()].sort_values("rank")
        return render.DataGrid(table, filters=True, height="620px")

    @reactive.calc
    def samples():
        frame = obs.iloc[filtered_indices()].copy()
        return frame.groupby(["GSE", "GSM"], observed=True).agg(
            tissue=("tissue", "first"), Disease=("Disease", "first"),
            Disease1=("Disease1", "first"), age=("age", "first"),
            gender=("gender", "first"), Cells=("GSM", "size"),
        ).reset_index()

    @output
    @render.data_frame
    def sample_table(): return render.DataGrid(samples(), filters=True, height="620px")

    @output
    @render.download(filename="filtered_atlas_samples.csv")
    def download_samples(): yield samples().to_csv(index=False).encode("utf-8-sig")


app = App(app_ui, server)
