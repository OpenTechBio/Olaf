import scrublet as scr
import numpy as np
import pandas as pd

def run_scrublet(
    adata,
    counts_layer='counts',    # Use 'counts' layer if available, else defaults to .X
    expected_doublet_rate=0.06,
    min_counts=2,
    min_cells=3,
    n_prin_comps=30,
    plot_hist=True,
    plot_umap=True,
    show=True,
    random_state=0
):
    """
    Runs Scrublet doublet detection on AnnData, annotates adata.obs.
    Returns: adata with 'doublet_score' and 'predicted_doublet' in .obs
    """
    # --- Extract counts matrix ---
    if counts_layer and counts_layer in adata.layers.keys():
        counts_matrix = adata.layers[counts_layer]
        print(f"Using layer '{counts_layer}' as raw counts for Scrublet.")
    else:
        counts_matrix = adata.X
        print("Using adata.X as raw counts for Scrublet (make sure this is not normalized/log1p).")
    
    if not isinstance(counts_matrix, np.ndarray):
        counts_matrix = counts_matrix.toarray()
    
    # --- Run Scrublet ---
    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=expected_doublet_rate, random_state=random_state)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=min_counts,
        min_cells=min_cells,
        n_prin_comps=n_prin_comps
    )

    # --- Add to AnnData ---
    adata.obs['doublet_score'] = doublet_scores
    adata.obs['predicted_doublet'] = pd.Categorical(predicted_doublets)
    
    # --- Plots ---
    if plot_hist:
        scrub.plot_histogram()
        if show: import matplotlib.pyplot as plt; plt.show()

    if plot_umap and 'X_umap' in adata.obsm:
        import scanpy as sc
        sc.pl.umap(adata, color=['doublet_score', 'predicted_doublet'], show=show)

    print(f"Scrublet completed: {predicted_doublets.sum()} predicted doublets out of {adata.n_obs} cells.")

    return adata

# Example usage:
# adata = run_scrublet(adata, counts_layer='counts')


def remove_doublets(adata, doublet_col='predicted_doublet', inplace=False):
    """
    Removes cells annotated as doublets from AnnData.
    
    Args:
        adata: AnnData object with doublet predictions in .obs
        doublet_col: obs column with doublet labels (default: 'predicted_doublet')
        inplace: if True, modifies the object in place. If False, returns a new object.
    Returns:
        Filtered AnnData object (if inplace=False)
    """
    n_before = adata.n_obs
    if doublet_col not in adata.obs:
        raise ValueError(f"{doublet_col} not found in adata.obs. Run doublet detection first.")
    mask = ~(adata.obs[doublet_col].astype(bool))
    n_doublets = (~mask).sum()
    print(f"Filtering out {n_doublets} predicted doublets of {n_before} cells.")
    if inplace:
        adata._inplace_subset_obs(mask)
        print(f"Remaining cells: {adata.n_obs}")
        return None
    else:
        filtered = adata[mask].copy()
        print(f"Remaining cells: {filtered.n_obs}")
        return filtered

# Example usage:
# adata = remove_doublets(adata)