import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models
from typing import Optional, Union

def train_celltypist_model(
    adata_ref,
    label_key: str,
    *,
    counts_layer: str = "counts",
    target_sum: float = 1e4,
    log1p: bool = True,
    feature_selection: bool = True,
    n_jobs: int = 8,
    use_SGD: Optional[bool] = None,
    max_iter: Optional[int] = None,
    model_out_path: Optional[str] = None
):
    """
    Train a CellTypist model on a labeled reference AnnData.

    - Expects labels in `adata_ref.obs[label_key]`.
    - Normalizes to `target_sum` and log1p on a copy.
    - If `model_out_path` is provided, saves the model (.pkl).
    """
    if label_key not in adata_ref.obs:
        raise KeyError(f"`{label_key}` not found in adata_ref.obs")

    ad = adata_ref.copy()
    if counts_layer in ad.layers:
        ad.X = ad.layers[counts_layer].copy()
    sc.pp.normalize_total(ad, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(ad)

    kwargs = dict(labels=label_key, feature_selection=feature_selection, n_jobs=n_jobs)
    if use_SGD is not None:
        kwargs["use_SGD"] = use_SGD
    if max_iter is not None:
        kwargs["max_iter"] = max_iter

    mdl = celltypist.train(ad, **kwargs)
    if model_out_path:
        mdl.write(model_out_path)
    return mdl


def annotate_with_model(
    adata,
    model_or_path: Union[str, "models.Model"],
    *,
    counts_layer: str = "counts",
    target_sum: float = 1e4,
    log1p: bool = True,
    obs_key: str = "celltypes",
    store_confidence: bool = True,
    majority_voting: bool = True
):
    """
    Annotate `adata` using a trained CellTypist model (object or path).

    - Normalizes/log1p on a working copy, writes labels to `adata.obs[obs_key]`.
    - Also stores confidence as `obs_key + '_conf'` (if available).
    """
    mdl = models.Model.load(model_or_path) if isinstance(model_or_path, str) else model_or_path

    ad = adata.copy()
    if counts_layer in ad.layers:
        ad.X = ad.layers[counts_layer].copy()
    sc.pp.normalize_total(ad, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(ad)

    preds = celltypist.annotate(ad, model=mdl, majority_voting=majority_voting)
    prad = preds.to_adata()  # holds 'majority_voting', 'predicted_labels', 'conf_score' in .obs

    label_col = "majority_voting" if majority_voting else "predicted_labels"
    labels = prad.obs[label_col].reindex(adata.obs_names)
    adata.obs[obs_key] = pd.Categorical(labels)

    if store_confidence and "conf_score" in prad.obs:
        adata.obs[obs_key + "_conf"] = prad.obs["conf_score"].reindex(adata.obs_names)

    return adata


# -----------------------
# Example (commented out)
# -----------------------
# # 1) Train on a labeled reference:
# model = train_celltypist_model(
#     adata_ref=adata_ref,
#     label_key="cell_type",
#     use_SGD=True,
#     n_jobs=16,
#     model_out_path="my_ref_model.pkl"  # optional
# )
#
# # 2) Annotate a query dataset with that model:
# adata = annotate_with_model(
#     adata=adata,
#     model_or_path=model,            # or "my_ref_model.pkl"
#     obs_key="celltypes"
# )
