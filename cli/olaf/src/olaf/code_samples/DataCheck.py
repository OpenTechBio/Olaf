import scanpy as sc
import pandas as pd
import numpy as np

def inspect_adata(adata, n_meta_preview=2):
    print("=== AnnData Quick Inspection ===")
    print(f"Shape: {adata.shape} (cells x genes)")
    print(f"obs keys: {list(adata.obs.columns)}")
    print(f"var keys: {list(adata.var.columns)}")
    print(f"layers: {list(adata.layers.keys())}")
    print(f"obsm: {list(adata.obsm.keys())}")
    print(f"uns: {list(adata.uns.keys())}")

    # --- Raw and counts layer checks ---
    has_raw = adata.raw is not None
    has_counts_layer = any("count" in l.lower() for l in adata.layers.keys())
    counts_layers = [l for l in adata.layers.keys() if "count" in l.lower()]
    print(f"Raw: {'Yes' if has_raw else 'No'} | Counts layer(s): {counts_layers if counts_layers else 'None'}")

    # --- Normalization check ---
    norm_layers = [l for l in adata.layers.keys() if "norm" in l.lower() or "log" in l.lower() or "scaled" in l.lower()]
    is_normalized = bool(norm_layers) or "log1p" in adata.uns
    print(f"Normalized: {'Yes' if is_normalized else 'No/Unknown'} | Normalization layers: {norm_layers if norm_layers else 'None'}")

    # --- Highly variable genes ---
    hvg_flagged = 'highly_variable' in adata.var.columns
    print(f"Highly variable genes flagged: {'Yes' if hvg_flagged else 'No'}")

    # --- Embeddings in obsm (with shapes) ---
    if len(adata.obsm.keys()) > 0:
        print("Embeddings in .obsm:")
        for k in adata.obsm.keys():
            shape = adata.obsm[k].shape
            if any(x in k.lower() for x in ['pca', 'umap', 'tsne', 'phate']):
                print(f"  {k} (shape {shape}) - Standard embedding")
            elif any(x in k.lower() for x in ['latent', 'harmony', 'scvi', 'mnn', 'scanorama', 'totalvi', 'cca']):
                print(f"  {k} (shape {shape}) - Likely integrated/latent space")
            else:
                print(f"  {k} (shape {shape}) - Unclassified, may be latent")
    else:
        print("No embeddings in .obsm")

    # --- Standard embeddings check ---
    print(f"PCA embedding: {'Yes' if 'X_pca' in adata.obsm else 'No'}")
    print(f"UMAP embedding: {'Yes' if 'X_umap' in adata.obsm else 'No'}")
    print(f"tSNE embedding: {'Yes' if 'X_tsne' in adata.obsm else 'No'}")

    # --- Common annotation keys (celltype, batch, doublet, leiden, sample, etc.) ---
    common_keys = ["batch", "celltype", "leiden", "doublet", "sample", "donor", "group", "predicted_labels"]
    found = [k for k in common_keys if k in adata.obs.columns]
    if found:
        print("Summary of common annotation keys:")
        for k in found:
            vals = adata.obs[k].unique()
            short = vals[:6] if len(vals) > 6 else vals
            print(f"  {k}: {short} (n={len(vals)})")
    else:
        print("No common annotation keys (batch, celltype, leiden, etc.) found.")

    # --- Null/Nan check for obs/var ---
    nulls_obs = adata.obs.isnull().sum().sum()
    nulls_var = adata.var.isnull().sum().sum()
    if nulls_obs or nulls_var:
        print(f"Warning: Missing values detected - obs: {nulls_obs}, var: {nulls_var}")

    # --- Sparse X check ---
    print(f"adata.X is sparse: {'Yes' if 'sparse' in str(type(adata.X)).lower() else 'No'}")

    # --- MultiIndex check ---
    if isinstance(adata.obs.index, pd.MultiIndex):
        print("WARNING: obs index is MultiIndex!")
    if isinstance(adata.var.index, pd.MultiIndex):
        print("WARNING: var index is MultiIndex!")

    # --- Preview metadata ---
    print("\nSample metadata (obs head):")
    print(adata.obs.head(n_meta_preview).T)

    print("=== End of Inspection ===")

# Example usage:
# adata = sc.read_h5ad("your_file.h5ad")
# inspect_adata(adata)
