import scanpy as sc
import celltypist
from celltypist import models

def list_celltypist_models():
    """List available CellTypist models."""
    available = models.models_overview()
    print("Available CellTypist models:")
    print(available[['name', 'description']])
    return available

def annotate_with_celltypist(
    adata,
    model_name=None,
    custom_markers=None,
    update_counts_layer=True,
    plot_umap=True,
    show=True
):
    """
    Annotate cells in AnnData using a CellTypist model or custom marker genes.

    Parameters:
        adata: AnnData object.
        model_name: Name or path of CellTypist model to use.
        custom_markers: Dict of manual marker gene sets, e.g.,
            {'CellTypeA': ['Gene1', 'Gene2'], ...}
        update_counts_layer: If True and 'counts' in adata.layers, set adata.X to raw counts.
        plot_umap: Plot UMAP colored by predicted cell type.
        show: Whether to immediately display plots.

    Returns:
        Annotated AnnData object with 'celltypes' column in .obs.
    """

    # Ensure raw counts are in .X for reproducibility
    if update_counts_layer and 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
        print("Reset .X to .layers['counts'] (raw counts).")

    # Normalize and log-transform if needed
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # CellTypist model-based annotation
    if model_name:
        print(f"Annotating with CellTypist model: {model_name}")
        # Download model if needed
        models.download_models(model=model_name, force_update=False)
        model = models.Model.load(model=model_name)
        # Run CellTypist
        predictions = celltypist.annotate(adata, model=model, majority_voting=True)
        adata.obs['celltypes'] = predictions.predicted_labels['majority_voting'].values
        print("CellTypist annotation complete.")
        print(adata.obs['celltypes'].value_counts())
    # Custom marker-based annotation
    elif custom_markers:
        print("Annotating using custom marker genes...")
        adata.obs['celltypes'] = 'Unknown'
        for cell_type, markers in custom_markers.items():
            valid = [g for g in markers if g in adata.var_names]
            if not valid:
                print(f"Warning: No valid markers for {cell_type}.")
                continue
            # Cells are positive if any of the marker genes are expressed (>0)
            expr = np.array(adata[:, valid].X)
            if expr.ndim > 1:
                expr = expr.sum(axis=1).A1 if hasattr(expr, 'A1') else expr.sum(axis=1).flatten()
            mask = expr > 0
            adata.obs.loc[mask, 'celltypes'] = cell_type
        print(adata.obs['celltypes'].value_counts())
    else:
        raise ValueError("Please specify either a model_name or custom_markers for annotation.")

    # Plot UMAP if available
    if plot_umap and 'X_umap' in adata.obsm:
        sc.pl.umap(adata, color='celltypes', title='Cell Type Annotation', show=show)
    return adata

# Example usage:
# List models
# list_celltypist_models()

# Annotate using built-in model
# adata = annotate_with_celltypist(adata, model_name='Immune_All_Low.pkl')

# Annotate using custom markers
# markers = {'T_cells': ['CD3D', 'CD3E'], 'B_cells': ['MS4A1'], 'Monocytes': ['CD14', 'LYZ']}
# adata = annotate_with_celltypist(adata, custom_markers=markers)
