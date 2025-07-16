# Don't import AutomMetric
# from AutoMetric import AutoMetric
import scanpy as sc
import celltypist
from celltypist import models
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import scanpy.external as sce

class CellTypingMetric(AutoMetric):
    """
    This is a class that computes cell typing using CellTypist
    Then, it evaluates using metrics from Benchmarker class from SCIB's Metric Module.
    """
    def metric(self, adata):
        #scib_metrics Benchmarker
        bm = Benchmarker(
            adata,
            batch_key="batch",
            label_key="majority_voting",
            bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_leiden=True),
            batch_correction_metrics=None,
            embedding_obsm_keys=["X_pca","X_pca_harmony"], #need to check if it has such a label -> if it doesnt perform pca
            n_jobs=6,
        )
        bm.prepare()
        bm.benchmark()
        bm.plot_results_table(min_max_scale=False)
        bm.get_results()

CellTypingMetric().run(adata)
