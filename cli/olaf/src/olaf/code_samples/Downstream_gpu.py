import os
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

# GPU / RAPIDS
import cupy as cp
import rapids_singlecell as rsc
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator


def _init_gpu():
    """
    Safe idempotent RMM init + CuPy allocator hookup.
    Call once per session.
    """
    try:
        # If already initialized, this is a no-op (rmm will raise only if incompatible)
        rmm.reinitialize(
            managed_memory=False,
            pool_allocator=False,
            devices=0,
        )
    except Exception:
        # If already initialized with different params, ignore.
        pass
    cp.cuda.set_allocator(rmm_cupy_allocator)


def standard_scanpy_downstream_gpu(
    adata,
    n_hvgs=2000,
    n_pcs=20,
    hvg_flavor="seurat_v3",  # rapids_singlecell uses its own implementation; flavor is kept for API parity
    random_state=0,
    save_layers=True,
    target_sum=1e4,
):
    """
    GPU-accelerated preprocessing & downstream using rapids_singlecell (rsc):
      - Ensure raw counts in .layers['counts']; start from it
      - Normalize (rsc.pp.normalize_total), save to .layers['normalized'] (GPU-backed, converted to CPU at the end)
      - log1p
      - HVGs (n_top_genes)
      - PCA (n_comps)
      - Neighbors, UMAP
      - Return with arrays on CPU

    Notes:
      * This function **restarts from counts** to ensure a clean, reproducible pipeline.
      * Results end on CPU (numpy/scipy) via rsc.get.anndata_to_CPU(adata).
    """
    # --- Ensure counts layer exists ---
    if save_layers and "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
        print("Raw counts saved to .layers['counts'].")

    # Always restart from counts for reproducibility
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        # If counts layer wasn't requested/saved previously, treat current X as counts
        adata.layers["counts"] = adata.X.copy()
        print("No .layers['counts'] found; using current X as counts and saving it.")

    # --- Init GPU / move AnnData to GPU ---
    _init_gpu()
    rsc.get.anndata_to_GPU(adata)

    # --- Normalize, save normalized (still on GPU, will be moved to CPU at the end) ---
    rsc.pp.normalize_total(adata, target_sum=target_sum)
    if save_layers:
        adata.layers["normalized"] = adata.X.copy()
        print("Normalized counts saved to .layers['normalized'].")

    # --- log1p ---
    rsc.pp.log1p(adata)

    # --- HVGs ---
    # rsc supports n_top_genes; flavor is kept for API parity but may be ignored internally
    rsc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)

    # --- PCA ---
    rsc.tl.pca(adata, n_comps=n_pcs, random_state=random_state)

    # --- Neighbors & UMAP ---
    # rsc.pp.neighbors will pick up PCA from .obsm['X_pca']
    rsc.pp.neighbors(adata, n_pcs=n_pcs)
    rsc.tl.umap(adata, random_state=random_state)

    # --- Bring everything back to CPU ---
    rsc.get.anndata_to_CPU(adata)

    print("GPU pipeline completed: counts -> normalize -> log1p -> HVGs -> PCA -> neighbors -> UMAP (returned on CPU).")
    return adata


# Example usage:
# adata = standard_scanpy_downstream_gpu(adata, n_hvgs=1000, n_pcs=50, random_state=0)
