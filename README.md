# Human CSF-PBMC Atlas: Connect Cloud bundle

This directory is a complete Shiny for Python application for Posit Connect
Cloud. 

## Data format

The `data/` directory is the GitHub-safe deployment representation of the
web-slim atlas. Every file is below GitHub's 100 MB single-file limit.

- `atlas_metadata.npz`: UMAP, categorical metadata codes, and four numeric
  cell-level QC arrays
- `metadata_categories.json`: category levels used to decode metadata
- `expression_part_*.npz`: CSC sparse matrices of quantized marker expression
- `manifest.json`: gene-to-file/column lookup and expression scale
- `cell_type_top50_marker_catalog.csv`: marker statistics

Expression values are restored as follows:

```python
matrix = scipy.sparse.load_npz("data/expression_part_00.npz")
logcounts = matrix[:, gene_column].toarray().ravel() / manifest["expression_scale"]
```

`cloud_data_report.json` records file sizes and verifies the GitHub limit.

The application supports discrete and QC-colored UMAPs, interactive legend
visibility controls, sample-level composition, marker expression, a marker
catalog, and violin/box plots for QC metrics or marker genes.
