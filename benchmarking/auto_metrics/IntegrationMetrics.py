# --- New metric class using scib-metrics ------------------------------------
from scib_metrics.benchmark import Benchmarker
from typing import Dict
import anndata
import numpy as np

EMBED = "X_scVI"        # The embedding key in adata.obsm
BATCH_KEY = "batch"     # The batch key in adata.obs
LABEL_KEY = "cell_type" # The cell type key in adata.obs

class IntegrationMetric(AutoMetric):
    """
    Compute SCIB integration quality metrics on an AnnData object using scib_metrics.
    Returns a dictionary with three metrics:
        • batch_silhouette: How well batches mix (lower ≈ better)
        • celltype_silhouette: How well cell types separate (higher ≈ better)
        • isolated_label_f1: Label preservation in isolated clusters (higher ≈ better)
    """
    def metric(self, adata):
        bm = Benchmarker(
            adata,
            batch_key=BATCH_KEY,
            label_key=LABEL_KEY,
            embedding_obsm_keys=[EMBED],        # list of embeddings to evaluate
        )
        bm.prepare()     # computes neighbors
        bm.benchmark()   # runs selected metrics
        results = bm.get_results()

        return results.to_dict()
    
IntegrationMetric().run(adata)