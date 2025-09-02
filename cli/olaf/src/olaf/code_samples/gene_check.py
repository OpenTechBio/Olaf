import re
import scanpy as sc
import pandas as pd
import numpy as np

# -------- helpers --------
_ENS_REGEX = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")  # Ensembl gene IDs (versioned or not)

def _strip_version(x: str) -> str:
    return x.split('.', 1)[0]

def _looks_like_ensembl(var_names: pd.Index) -> float:
    """Return fraction of var_names that look like Ensembl gene IDs."""
    return np.mean([bool(_ENS_REGEX.match(str(v))) for v in var_names])

def _guess_species_from_ens(var_names: pd.Index) -> str:
    """Very rough guess from Ensembl prefix."""
    base = [str(v) for v in var_names[: min(2000, len(var_names))]]
    counts = {
        "human": sum(s.startswith("ENSG") for s in base),
        "mouse": sum(s.startswith("ENSMUSG") for s in base),
        "zebrafish": sum(s.startswith("ENSDARG") for s in base),
        "rat": sum(s.startswith("ENSRNOG") for s in base),
        "cow": sum(s.startswith("ENSBTAG") for s in base),
        "pig": sum(s.startswith("ENSSSCG") for s in base),
        "chicken": sum(s.startswith("ENSGALG") for s in base),
        "macaque": sum(s.startswith("ENSMMUG") for s in base),
    }
    sp, n = max(counts.items(), key=lambda kv: kv[1])
    return sp if n > 0 else "unknown"

def _pick_symbol_column(var_df: pd.DataFrame) -> str | None:
    """Pick a likely gene-symbol column if present."""
    candidates = [
        "gene_symbols","gene_symbol","gene_name","gene_names","feature_name",
        "Symbol","SYMBOL","symbol","gene","name"
    ]
    for c in candidates:
        if c in var_df.columns:
            # require at least some non-null strings
            if var_df[c].astype(str).str.len().gt(0).sum() >= 0.5 * len(var_df):
                return c
    return None

def _make_unique(names: pd.Series, fallback: pd.Index) -> pd.Index:
    """Ensure names are unique; if duplicates, suffix with _2, _3, or add Ensembl fallback."""
    seen = {}
    out = []
    for sym, ens in zip(names.astype(str).tolist(), fallback.tolist()):
        base = sym if sym not in ("", "nan", "None") else ens
        if base in seen:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 1
            out.append(base)
    return pd.Index(out)

def rename_var_from_ensembl_to_symbols(
    adata,
    species: str | None = None,
    in_place: bool = True,
    prefer_column: str | None = None,
):
    """
    If var_names are Ensembl-like, rename them to gene symbols using an existing
    column in adata.var (e.g., gene_symbols/gene_name/feature_name). Preserves
    original IDs in adata.var['ensembl_id'] and makes names unique.

    Parameters
    ----------
    species : optional str
        Only used for logging/your awareness. (If you later enable mygene,
        you'd use this; implementation below avoids web calls.)
    in_place : bool
        Modify adata in place.
    prefer_column : str
        If you know the exact column holding symbols, provide it.
    """
    frac_ens = _looks_like_ensembl(adata.var_names)
    if frac_ens < 0.80:
        print(f"[rename] var_names do not look like Ensembl IDs (ENS fraction={frac_ens:.2f}). Skipping.")
        return adata

    sp_guess = _guess_species_from_ens(adata.var_names)
    if species is None:
        species = sp_guess
    print(f"[rename] Detected Ensembl-like var_names (ENS fraction={frac_ens:.2f}). Species guess: {species}")

    # keep original Ensembl IDs (strip version for clarity)
    if "ensembl_id" not in adata.var.columns:
        adata.var["ensembl_id"] = pd.Index(adata.var_names).map(_strip_version).values

    # choose the symbol column
    col = prefer_column or _pick_symbol_column(adata.var)
    if col is None:
        print("[rename] No obvious gene-symbol column found in adata.var. "
              "If you have symbols, put them in one of: "
              "gene_symbols/gene_symbol/gene_name/feature_name/symbol/name.")
        # Optionally: you could add a mygene-based mapping here if you want internet queries.
        return adata

    print(f"[rename] Using symbol column: '{col}'")
    # If there's a matching Ensembl column, align by it; otherwise assume row alignment
    ens_col_candidates = ["gene_ids","gene_id","ensembl","ensembl_id"]
    ens_col = next((c for c in ens_col_candidates if c in adata.var.columns), None)

    # Prepare new names
    if ens_col is not None:
        ens_col_stripped = adata.var[ens_col].astype(str).map(_strip_version)
        var_names_stripped = pd.Index(adata.var_names).map(_strip_version)
        if ens_col_stripped.equals(var_names_stripped):
            # 1:1 aligned
            new_names_raw = adata.var[col].astype(str)
        else:
            # Align by Ensembl id
            mapper = pd.Series(adata.var[col].astype(str).values, index=ens_col_stripped).to_dict()
            new_names_raw = pd.Index(var_names_stripped).map(lambda k: mapper.get(k, k))
    else:
        # assume row-wise alignment
        new_names_raw = adata.var[col].astype(str)

    # Ensure uniqueness and sensible fallbacks
    new_names = _make_unique(pd.Series(new_names_raw.values), fallback=adata.var["ensembl_id"])

    # Report a small preview
    print("[rename] Example before -> after:")
    for old, new in zip(adata.var_names[:5], new_names[:5]):
        print(f"  {old} -> {new}")

    if in_place:
        adata.var_names = new_names
        # Keep a convenience column too
        adata.var["gene_symbol"] = new_names.values
        print(f"[rename] Completed: var_names replaced (n={adata.n_vars}).")
        return adata
    else:
        ad = adata.copy()
        ad.var_names = new_names
        ad.var["gene_symbol"] = new_names.values
        print(f"[rename] Completed on a copy: var_names replaced (n={ad.n_vars}).")
        return ad
