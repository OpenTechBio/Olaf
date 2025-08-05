import scanpy as sc

def run_harmony(
    adata,
    batch_key,
    n_pcs=20,
    n_neighbors=15,
    leiden_resolution=1.0,
    run_umap=True,
    run_leiden=True,
    plot_umap=True,
    show=True
):
    """
    Runs Harmony batch correction on AnnData PCA embeddings.
    Adds corrected PCA to .obsm['X_pca_harmony'].
    Optionally recomputes neighbors, UMAP, and Leiden on the corrected space.

    Args:
        adata: AnnData object (must have .obsm['X_pca'])
        batch_key: obs column to use as batch variable
        n_pcs: number of PCs to use
        n_neighbors: neighbors for graph construction
        leiden_resolution: Leiden resolution
        run_umap: compute UMAP on Harmony space
        run_leiden: run Leiden clustering on Harmony space
        plot_umap: plot UMAP by batch and cluster
        show: display plots

    Returns:
        AnnData object with corrected embeddings and new clustering/UMAP (if run).
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError("You need to install harmonypy: pip install harmonypy")

    # Ensure PCA is computed
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=n_pcs)

    # Run Harmony
    import scanpy.external as sce
    print(f"Running Harmony on {n_pcs} PCs with batch key '{batch_key}'...")
    sce.pp.harmony_integrate(adata, key=batch_key, basis='X_pca')

    # Save harmony embedding
    if 'X_pca_harmony' in adata.obsm:
        print("Harmony-corrected PCA saved to .obsm['X_pca_harmony']")
    else:
        raise RuntimeError("Harmony did not produce 'X_pca_harmony' in .obsm.")

    # Downstream: neighbors, UMAP, Leiden
    if run_umap or run_leiden:
        sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_pcs=n_pcs, n_neighbors=n_neighbors)
        if run_umap:
            sc.tl.umap(adata)
        if run_leiden:
            sc.tl.leiden(adata, resolution=leiden_resolution, key_added='leiden_harmony')
    
    # Plot
    if plot_umap and run_umap:
        sc.pl.umap(adata, color=[batch_key, 'leiden_harmony' if run_leiden else None], show=show)

    return adata

# Example usage:
# adata = run_harmony(adata, batch_key='batch', n_pcs=20)
