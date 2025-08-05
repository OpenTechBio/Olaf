import scanpy as sc
import numpy as np

def full_reprocess_and_plot(
    adata,
    n_hvgs=2000,
    n_pcs=20,
    leiden_resolution=1.0,
    hvg_flavor='seurat',
    cluster_key='leiden',
    save_layers=True,
    umap_color=None,    # None = use cluster_key
    random_state=0,
    plot_show=True
):
    """
    Revert to raw counts, re-normalize, and rerun full downstream analysis.
    Plots UMAP and top DGE dotplot.
    """
    # --- Revert to raw counts ---
    if "counts" not in adata.layers:
        raise ValueError("No .layers['counts'] found—can't revert to raw counts!")
    adata.X = adata.layers["counts"].copy()
    print("Reverted .X to raw counts from .layers['counts'].")

    # --- Normalize total counts ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    if save_layers:
        adata.layers["normalized"] = adata.X.copy()
        print("Overwrote .layers['normalized'].")

    # --- Log1p transform ---
    sc.pp.log1p(adata)

    # --- Highly variable genes ---
    sc.pp.highly_variable_genes(
        adata, flavor=hvg_flavor, n_top_genes=n_hvgs
    )

    # --- PCA ---
    sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)

    # --- Neighbors ---
    sc.pp.neighbors(adata, n_pcs=n_pcs)

    # --- UMAP ---
    sc.tl.umap(adata, random_state=random_state)

    # --- Leiden clustering ---
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added=cluster_key, random_state=random_state)

    # --- Plot UMAP by cluster ---
    umap_color = umap_color or cluster_key
    sc.pl.umap(adata, color=umap_color, show=plot_show)

    # --- DGE and Dotplot ---
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=3, groupby=cluster_key, show=plot_show)

    print("Processing and plotting completed.")

    return adata

# Example usage:
# adata = sc.read_h5ad('your_file.h5ad')
# adata = full_reprocess_and_plot(adata)