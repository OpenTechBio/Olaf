import numpy as np
from scipy.stats import median_abs_deviation

def is_outlier(adata, metric: str, nmads: float = 5):
    """
    Returns a boolean Series marking outliers for a given metric.
    Outlier is defined as more than nmads from the median.
    """
    M = adata.obs[metric]
    med = np.median(M)
    mad = median_abs_deviation(M, scale='normal')  # robust to outliers, uses standard scaling
    outlier = (M < med - nmads * mad) | (M > med + nmads * mad)
    return outlier

# General QC outliers (use your preferred metrics)
adata.obs["outlier"] = (
    is_outlier(adata, "log1p_total_counts", 5)
    | is_outlier(adata, "log1p_n_genes_by_counts", 5)
    | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
)
print("QC outlier value counts:")
print(adata.obs["outlier"].value_counts())

# Mitochondrial outliers (can combine MAD with a hard threshold)
adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (adata.obs["pct_counts_mt"] > 8)
print("Mitochondrial outlier value counts:")
print(adata.obs["mt_outlier"].value_counts())

print(f"Total number of cells before filtering: {adata.n_obs}")

# Apply filters
adata = adata[(~adata.obs["outlier"]) & (~adata.obs["mt_outlier"])].copy()

print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")