# Don't import AutomMetric
# from AutoMetric import AutoMetric 
import scanpy as sc

class CellCountMetric(AutoMetric):
    """
    A simple metric to count the number of cells and genes.
    """
    def metric(self, adata) -> dict:
        num_cells = adata.n_obs
        num_genes = adata.n_vars
        
        return {
            "Number of Cells": num_cells,
            "Number of Genes": num_genes
        }
    
# must run it here
CellCountMetric().run(adata)