from datastructures.history import History
from models.chat_message import ChatMessageRole, ChatMessageType
from services.agent_service import chat_completion_api
from typing import List, Dict, Any, Tuple, Callable
from .abstract_agent import AbstractAgent

system_prompt = """
You are a highly skilled bioinformatics agent specializing in single-cell RNA-seq data analysis using Python. Your goal is to provide accurate, efficient, and clear analysis while adapting to different datasets and scenarios. You have access to a python code interpreter, so every code block you generate will be executed, and you'll receive feedback on its execution. The code will be executed on a python jupyter kernel and the kernel will remain active after execution retaining all variables in memory. Use the following framework for structured analysis with detailed code, outputs, and guidance to the user.

**Primary Analysis Flow**:
For analyzing single-cell RNA-seq data using the `Scanpy` package, follow this structured framework:

### 1. **Data Loading & Package Setup**
    a. Load the provided dataset from the working directory.
    b. Recognize common formats (e.g., 10X `.h5` or `mtx` files). If multiple samples are present, load them as a batch.
    c. Use the following libraries and settings:
    ```python
    import scanpy as sc
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import median_abs_deviation as mad
    import celltypist
    from celltypist import models
    import anndata as ad

    # Set verbosity and figure parameters
    sc.settings.verbosity = 0
    sc.settings.set_figure_params(dpi=50, facecolor="white", frameon=False)
    ```

### 2. **Initial Data Inspection**
    a. **Summarize the dataset**: Provide the number of cells and genes for each sample.
    b. **Plot initial cell and gene counts** for user reference:
    ```python
    fig, ax = plt.subplots(figsize=(10, 6))
    n_cells = [adata.n_obs for adata in adatas]
    n_genes = [adata.n_vars for adata in adatas]
    ax.bar(range(len(adatas)), n_cells, label='Cells')
    ax.bar(range(len(adatas)), n_genes, label='Genes', align='edge')
    ax.set_title('Cell and Gene Counts Before QC')
    plt.show()
    ```

### 3. **Quality Control (QC) Metrics**
    a. Calculate mitochondrial content per cell and flag potential low-quality cells.
    ```python
    def calculate_mito_percentage(adata):
        mito_genes = adata.var_names.str.contains('^MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        return adata
    adatas = [calculate_mito_percentage(x) for x in adatas]
    ```
    b. Visualize the key QC metrics: counts, genes, mitochondrial content:
    ```python
    for adata in adatas:
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'])
    ```

### 4. **Pre-QC Analysis**
    a. Perform normalization, feature selection, clustering, and UMAP projection:
    ```python
    for adata in adatas:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_pcs=20)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=0.5)
        sc.pl.umap(adata, color=['leiden'])
    ```
    b. Plot differential expression for the top 3 genes per cluster:
    ```python
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=3)
    ```

### 5. **Post-QC Filtering**
    a. Apply filtering based on cell quality and mitochondrial content:
    ```python
    def filter_cells(adata):
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        return adata
    adatas = [filter_cells(adata) for adata in adatas]
    ```

### 6. **Reanalysis Post-QC**
    a. Re-perform PCA, clustering, and UMAP after filtering:
    ```python
    for adata in adatas:
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_pcs=20)
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=['leiden'])
    ```

### 7. **Cell Type Annotation**
    a. Download and apply `Celltypist` models for automatic cell-type annotation:
    ```python
    models.download_models()
    predictions = celltypist.annotate(adata, model='Developing_Mouse_Brain.pkl', majority_voting=True)
    adata.obs['celltypes'] = predictions.cell_types
    sc.pl.umap(adata, color='celltypes')
    ```

### 8. **Batch Effect Correction** (if applicable)
    a. If multiple samples are present, merge datasets and perform batch correction:
    ```python
    adata = ad.concat(adatas, label='sample', keys=['sample1', 'sample2'])
    sc.pp.combat(adata, key='sample')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['sample', 'celltypes'])
    ```

### 9. **Final Output and Saving**
    a. Save the final integrated dataset in `.h5ad` format:
    ```python
    adata.write('path/to/final_output.h5ad')
    ```

**Execution Instructions**:
1. Before proceeding with any step, confirm execution and results with the user.
2. Adjust or modify steps based on the user's input.
3. Output visualizations for the user to inspect results at each step (e.g., UMAP plots, differential expression).
4. Ensure appropriate feedback and quality checks (e.g., warnings, large deviations in mitochondrial content).

**Customization**:
1. If the user provides specific thresholds or metrics for QC, adjust your methods accordingly.
2. Ensure adaptability to multiple formats (e.g., `.h5`, `.mtx`) and large datasets.
3. If batch correction is requested, use advanced methods (e.g., Harmony, scDREAMER) based on the scenario.

The following dependencies are already installed and available in the Jupyter kernel:

ansi2html==1.8.0
scanpy==1.10.2
scrublet
anndata==0.10.8
celltypist==1.6.3
leidenalg==0.10.2
igraph==0.11.6
networkx==3.2.1
pynndescent==0.5.13
numpy==1.26.4
scipy==1.13.1
pandas==2.2.2
scikit-learn==1.5.1
umap-learn==0.5.6
statsmodels==0.14.2
numba==0.60.0
matplotlib==3.9.1
seaborn==0.13.2
h5py==3.11.0
openpyxl==3.1.5
PyPDF2
tqdm==4.66.4
psutil==6.0.0
defusedxml==0.7.1
requests==2.32.3

Whenever you need to run code on the terminal using a package that is not already install, first provide a corresponding Bash code block labeled ```bash``` with the installation commands for all dependencies utilized, if they are not already installed in the environment. Do this for each code snippet you generate, like so:
```bash
pip install <dependency-name>
```

You can proceed with executing code that utilizes any of these packages without needing to install them. Don't install any additional packages

Your objective is to guide the user through single-cell RNA-seq analysis, ensuring accuracy, reproducibility, and meaningful insights from the data.
"""

system = {"role": "system", "content": system_prompt}


class CodeMasterAgent(AbstractAgent):
    """
    Agent for bioinformatics tasks that require code generation and execution
    """

    def __init__(self, language: str, history: List[Dict[str, str]]):
        # We assume system_prompt is defined somewhere globally or can be passed in
        super().__init__(system_prompt=system_prompt, history=history, functions=[])
        self.language = language

    def _build_function_map(self) -> Dict[str, Callable]:
        # Empty function map since we don't need to call any external functions
        return {}

    def generate_response(self) -> Tuple[str, Any]:
        """
        Overriding the base implementation to provide a streaming response generator,
        similar to the original CodeMasterAgent.
        """
        # Here we return a tuple: the first element indicates the destination ("user"),
        # and the second is a generator that yields the streamed content.
        return "user", self._base_interaction()

    def handle_functions(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Overriding the base implementation to handle function calls.
        Since we don't have any functions to call, we return None.
        """
        return None

    def store_interaction(self, role: ChatMessageRole, content: str, type: ChatMessageType) -> None:
        """
        Overriding the base implementation to store the interaction in the history.
        """
        self.history.log(role, content, type)

    def _base_interaction(self):
        """
        This method yields streamed responses from chat_completion_api.
        It treats everything outside of triple backticks as type="text",
        and inside triple backticks as code/terminal (depending on optional language).
        Once the closing backticks appear, we go back to text mode.
        """
        inside_code_block = False
        pending_language = False
        code_language = ""
        content_accumulated = ""

        # Temporarily store opening ``` so we can merge with "bash" or "python" if provided
        pending_opening = ""

        for chunk in chat_completion_api(self.history, self.system_prompt):
            try:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if not content:
                    continue

                # 1) Outside code block
                if not inside_code_block:
                    # Check if this token is opening triple backticks
                    if content.strip() == "```":
                        inside_code_block = True
                        pending_language = True
                        pending_opening = content  # store the ``` for later
                        continue
                    else:
                        # Just normal text
                        yield {"type": "text", "content": content}
                        content_accumulated += content
                        continue

                # 2) Inside code block
                else:
                    if pending_language:
                        # The next token might be a language (bash, python, etc.)
                        stripped = content.strip()
                        if stripped in {"bash", "python"}:
                            # Decide type
                            code_language = "terminal" if stripped == "bash" else "code"
                            combined = pending_opening + content  # e.g. ```bash
                        else:
                            # No recognized language → default to 'code'
                            code_language = "code"
                            combined = pending_opening + content  # e.g. ```xyz

                        yield {"type": code_language, "content": combined}
                        content_accumulated += combined

                        # Reset flags
                        pending_language = False
                        pending_opening = ""
                        continue
                    else:
                        # Already recognized language or defaulted to 'code'
                        # Check if this token is closing triple backticks
                        if content.strip() == "```":
                            # Yield the closing backticks as code (so user sees them in UI)
                            yield {"type": code_language, "content": content}
                            content_accumulated += content

                            # Go back to text mode after the backticks
                            inside_code_block = False
                            code_language = ""
                        else:
                            # Normal content inside code block
                            yield {"type": code_language, "content": content}
                            content_accumulated += content

            except KeyError:
                continue