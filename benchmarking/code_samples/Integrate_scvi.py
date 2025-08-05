import scanpy as sc

def run_scvi_integration(
    adata,
    batch_key=None,
    n_latent=30,
    n_neighbors=15,
    leiden_resolution=1.0,
    run_umap=True,
    run_leiden=True,
    plot_umap=True,
    show=True,
    max_epochs=50,
    use_gpu=True
):
    """
    Runs scVI integration and computes downstream embedding/clustering.

    Args:
        adata: AnnData object (must have raw counts in .X or .layers['counts'])
        batch_key: obs column for batch
        n_latent: number of latent dimensions
        n_neighbors: for neighbor graph
        leiden_resolution: for Leiden clustering
        run_umap: compute UMAP on scVI latent space
        run_leiden: run Leiden clustering on scVI space
        plot_umap: plot UMAP colored by batch and Leiden
        show: whether to display plots
        max_epochs: epochs for training (increase for large datasets)
        use_gpu: set False to force CPU
    Returns:
        AnnData with .obsm["X_scVI"] and downstream clustering.
    """
    try:
        import scvi
    except ImportError:
        raise ImportError("Install scvi-tools: pip install scvi-tools")

    # Setup AnnData for scVI
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)

    # Train scVI
    vae = scvi.model.SCVI(adata, n_latent=n_latent)
    vae.train(max_epochs=max_epochs, use_gpu=use_gpu)
    print("scVI model trained.")

    # Get latent representation
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    print("scVI latent space saved to .obsm['X_scVI'].")

    # Downstream: neighbors, UMAP, Leiden
    if run_umap or run_leiden:
        sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=n_neighbors)
        if run_umap:
            sc.tl.umap(adata)
        if run_leiden:
            sc.tl.leiden(adata, resolution=leiden_resolution, key_added="leiden_scvi")

    # Plot UMAP
    if plot_umap and run_umap:
        sc.pl.umap(adata, color=[batch_key, "leiden_scvi" if run_leiden else None], show=show)

    return adata

# Example usage:
# adata = run_scvi_integration(adata, batch_key="batch", n_latent=30)
