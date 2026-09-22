# Human CSF-PBMC Atlas: Connect Cloud bundle

This directory is a complete Shiny for Python application for Posit Connect
Cloud. Upload **this directory only** as a standalone GitHub repository. Do not
upload the parent project, which contains the 38 GB source H5AD.

## Test locally

```bash
cd /home/lishaogang/NEU2026_July/final/ShinyAppsCloud
/home/lishaogang/miniconda3/envs/sc_viewer/bin/shiny run \
  --host 127.0.0.1 --port 8050 app.py
```

## Publish through GitHub

Create an empty GitHub repository, then run these commands after replacing the
account and repository names:

```bash
cd /home/lishaogang/NEU2026_July/final/ShinyAppsCloud
git init
git add app.py requirements.txt README.md .gitignore www data cloud_data_report.json
git commit -m "Publish CSF-PBMC single-cell atlas"
git branch -M main
git remote add origin https://github.com/GITHUB_ACCOUNT/REPOSITORY.git
git push -u origin main
```

In Posit Connect Cloud:

1. Sign in at <https://connect.posit.cloud/>.
2. Click **Publish** and install/authorize the GitHub App if prompted.
3. Select **Shiny for Python**, this repository, and branch `main`.
4. Select `app.py` as the primary file.
5. Select Python 3.10 and leave automatic republishing enabled.
6. Publish, then set a stable custom content URL in the content settings.

No secret variable is required for this application. On the Free plan the
application and its source repository are public.

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
