import scanpy as sc
import matplotlib.pyplot as plt

def plot_qc_on_embeddings(
    adata,
    qc_keys = ["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_in_top_20_genes"],
    embeddings = ["X_umap", "X_pca"],
    show=True
):
    """
    Plots selected QC metrics on PCA and UMAP embeddings.
    Args:
        adata: AnnData object (must have UMAP/PCA computed)
        qc_keys: list of obs keys to plot
        embeddings: which embeddings to use
        show: whether to display plots immediately
    """
    for emb in embeddings:
        if emb not in adata.obsm.keys():
            print(f"Warning: {emb} not found in adata.obsm, skipping.")
            continue
        for key in qc_keys:
            if key in adata.obs.columns:
                sc.pl.embedding(adata, basis=emb.replace("X_", ""), color=key, title=f"{key} on {emb.replace('X_', '')}", show=show)
            else:
                print(f"QC metric {key} not found in adata.obs, skipping.")

# Example usage:
# plot_qc_on_embeddings(adata)
