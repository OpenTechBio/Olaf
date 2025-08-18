import scanpy as sc
import numpy as np

def standard_scanpy_downstream(
    adata,
    n_hvgs=2000,
    n_pcs=20,
    hvg_flavor = 'seurat',
    random_state=0,
    save_layers=True,
):
    """
    Full Scanpy preprocessing and downstream pipeline:
      - Save raw counts to .layers['counts']
      - Normalize, save normalized to .layers['normalized']
      - log1p
      - Find HVGs (Seurat v3)
      - PCA, neighbors, UMAP, Leiden
    Args:
      adata: AnnData object
      hvg_flavor: method for HVG selection
      n_hvgs: number of HVGs
      n_pcs: number of PCs for PCA/neighbors/UMAP
      leiden_resolution: clustering resolution
      random_state: for reproducibility
      save_layers: whether to save layers for counts and normalized data
    Returns:
      AnnData (modified in-place)
    """
    # --- Save raw counts if not already present ---
    if save_layers and "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
        print("Raw counts saved to .layers['counts'].")

    # --- Check for prior normalization ---
    already_normalized = (
        (save_layers and "normalized" in adata.layers)
        or ("normalized" in adata.layers)
        or ("normalize_total" in adata.uns)
    )
    if already_normalized:
        print("Skipping normalization: normalized counts already present.")
        if save_layers:
            adata.X = adata.layers["normalized"].copy()
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        if save_layers:
            adata.layers["normalized"] = adata.X.copy()
            print("Normalized counts saved to .layers['normalized'].")

    # --- Check for prior log1p transformation ---
    # This is tricky; check if adata.uns has 'log1p' or if all X < 20 (as a heuristic)
    already_logged = (
        "log1p" in adata.uns
        or (np.max(adata.X) < 20 and np.mean(adata.X) < 5)
    )
    if already_logged:
        print("Skipping log1p: appears to already be log-transformed.")
    else:
        sc.pp.log1p(adata)

    # --- Highly variable genes (HVGs) ---
    if 'highly_variable' in adata.var.columns and adata.var['highly_variable'].sum() >= n_hvgs:
        print("Skipping HVG selection: already present.")
    else:
        sc.pp.highly_variable_genes(
            adata, flavor=hvg_flavor, n_top_genes=n_hvgs
        )

    # --- PCA ---
    if "X_pca" in adata.obsm:
        print("Skipping PCA: already present.")
    else:
        sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)

    # --- Neighbors ---
    if "neighbors" in adata.uns:
        print("Skipping neighbors: already computed.")
    else:
        sc.pp.neighbors(adata, n_pcs=n_pcs)

    # --- UMAP ---
    if "X_umap" in adata.obsm:
        print("Skipping UMAP: already present.")
    else:
        sc.tl.umap(adata, random_state=random_state)

    print("Pipeline completed: counts, normalized, log1p, HVGs, PCA, neighbors, UMAP.")
    return adata

# Example usage:
# adata = standard_scanpy_downstream(adata)