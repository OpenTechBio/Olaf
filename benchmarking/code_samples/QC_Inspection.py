import scanpy as sc
import numpy as np
import pandas as pd

def calculate_and_plot_qc(
    adata,
    mito_prefix="mt-",
    ribo_prefix=None,  # Set to None if you don't want ribo
    make_plots=True,
    show=True
):
    """
    Calculates and visualizes standard QC metrics:
      - n_genes_by_counts, total_counts, pct_counts_mt, pct_counts_ribo (optional)
      - log1p metrics
      - percent_top_20
    Args:
        adata: AnnData object
        mito_prefix: prefix for mitochondrial genes
        ribo_prefix: prefix for ribosomal protein genes (optional)
        make_plots: if True, create violin/scatter plots
        show: if True, show plots immediately
    Returns:
        adata (modified), qc_metrics DataFrame
    """

    # --- Tag mitochondrial genes ---
    adata.var['mt'] = adata.var_names.str.startswith(mito_prefix)

    # --- Optionally tag ribosomal genes ---
    if ribo_prefix:
        adata.var['ribo'] = adata.var_names.str.startswith(ribo_prefix)
        qc_vars = ['mt', 'ribo']
    else:
        qc_vars = ['mt']

    # --- Calculate QC metrics ---
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        percent_top=[20],
        log1p=True,        # <--- Calculate log1p metrics
        inplace=True
    )

    # --- Compose QC metrics DataFrame ---
    metrics = [
        'n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'log1p_n_genes_by_counts',
        'log1p_total_counts', 'pct_counts_in_top_20_genes'
    ]
    if ribo_prefix and 'pct_counts_ribo' in adata.obs.columns:
        metrics += ['pct_counts_ribo']
    if ribo_prefix and 'log1p_pct_counts_ribo' in adata.obs.columns:
        metrics += ['log1p_pct_counts_ribo']

    qc_metrics = adata.obs[[k for k in metrics if k in adata.obs.columns]].copy()

    # --- Plots ---
    if make_plots:
        # Violin plots for main metrics and percent_top_20
        plot_keys = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_in_top_20_genes']
        if ribo_prefix and 'pct_counts_ribo' in adata.obs.columns:
            plot_keys.append('pct_counts_ribo')
        sc.pl.violin(adata, plot_keys, jitter=0.4, multi_panel=True, show=show)

        # Scatter plots
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=show)
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', show=show)
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_in_top_20_genes', show=show)

    return adata, qc_metrics

# Example usage:
# adata, qc_metrics = calculate_and_plot_qc(adata, mito_prefix='mt-', ribo_prefix='Rps')

