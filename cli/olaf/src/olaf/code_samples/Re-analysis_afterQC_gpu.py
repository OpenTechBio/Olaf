import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

# RAPIDS / GPU
import cupy as cp
import rapids_singlecell as rsc
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator


def _init_gpu():
    """Idempotent RMM init and CuPy allocator hookup."""
    try:
        rmm.reinitialize(
            managed_memory=False,   # set True if you want oversubscription
            pool_allocator=False,   # keep False unless you want a memory pool
            devices=0,              # register GPU 0
        )
    except Exception:
        pass
    cp.cuda.set_allocator(rmm_cupy_allocator)


def full_reprocess_and_plot_gpu(
    adata,
    n_hvgs=2000,
    n_pcs=20,
    leiden_resolution=1.0,
    hvg_flavor='seurat',      # kept for API parity; rsc uses its own HVG impl
    cluster_key='leiden',
    save_layers=True,
    umap_color=None,
    random_state=0,
    plot_show=True,
    target_sum=1e4,
):
    """
    GPU-accelerated reprocess + plots (using rapids_singlecell on GPU, then CPU plots):

    Steps on GPU:
      - Revert .X to counts
      - rsc.pp.normalize_total (save to .layers['normalized'])
      - rsc.pp.log1p
      - rsc.pp.highly_variable_genes(n_top_genes=n_hvgs)
      - rsc.tl.pca(n_comps=n_pcs)
      - rsc.pp.neighbors(n_pcs=n_pcs)
      - rsc.tl.umap(random_state)
      - rsc.tl.leiden(resolution, key_added=cluster_key)

    Then:
      - Bring AnnData back to CPU
      - Plot UMAP, run DGE (wilcoxon) on CPU, and dotplot top markers.

    Returns:
      AnnData on CPU with UMAP, Leiden, and DGE results.
    """
    # --- Revert to raw counts ---
    if "counts" not in adata.layers:
        raise ValueError("No .layers['counts'] found—can't revert to raw counts!")
    adata.X = adata.layers["counts"].copy()
    print("Reverted .X to raw counts from .layers['counts'].")

    # --- Initialize GPU + move AnnData to GPU ---
    _init_gpu()
    rsc.get.anndata_to_GPU(adata)

    # --- Normalize (save normalized layer) ---
    rsc.pp.normalize_total(adata, target_sum=target_sum)
    if save_layers:
        adata.layers["normalized"] = adata.X.copy()
        print("Overwrote .layers['normalized'] on GPU (will be CPU after transfer).")

    # --- log1p ---
    rsc.pp.log1p(adata)

    # --- HVGs ---
    rsc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)

    # --- PCA ---
    rsc.tl.pca(adata, n_comps=n_pcs, random_state=random_state)

    # --- Neighbors ---
    rsc.pp.neighbors(adata, n_pcs=n_pcs)

    # --- UMAP ---
    rsc.tl.umap(adata, random_state=random_state)

    # --- Leiden clustering (on GPU) ---
    rsc.tl.leiden(adata, resolution=leiden_resolution, key_added=cluster_key, random_state=random_state)

    # --- Bring back to CPU for Scanpy plotting & DGE ---
    rsc.get.anndata_to_CPU(adata)

    # --- Plot UMAP by cluster (CPU) ---
    umap_color = umap_color or cluster_key
    sc.pl.umap(adata, color=umap_color, show=plot_show)

    # --- DGE (CPU) + Dotplot ---
    rsc.tl.rank_genes_groups_logreg(adata, groupby=cluster_key, use_raw=False)
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=3, groupby=cluster_key, show=plot_show)

    print("GPU reprocessing + plotting completed.")
    return adata

# Example usage:
# adata = full_reprocess_and_plot_gpu(adata, n_hvgs=2000, n_pcs=30, leiden_resolution=0.8, random_state=0)
