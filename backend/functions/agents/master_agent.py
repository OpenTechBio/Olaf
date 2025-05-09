### ----------------------------------------------
### THIS AGENT IMPLMENTATION IS OUTDATED
### PLEASE REFER TO THE ABSTRACT AGENT IMPLEMENTATION
### FOR THE MOST UP-TO-DATE STANDARD
### ----------------------------------------------
from services.agent_service import extract_code_and_text, chat_completion_api, chat_completion_plan, chat_completion_function
import openai
import json
import re

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

You can proceed with executing code that utilizes any of these packages without needing to install them. Don't install any additional packages

Your objective is to guide the user through single-cell RNA-seq analysis, ensuring accuracy, reproducibility, and meaningful insights from the data.
"""


system = {
    "role": "system",
    "content": system_prompt
}


class MasterAgent:
    def __init__(self,history):
        self.system_prompt = system_prompt
        self.history = history
        self.functions = [
            {
                "type": "function",
                "function": {
                    "name": "handle_simple_interaction",
                    "description": "Handles simple user interactions. This is for like human interactions or something like that.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to process."
                            },
                            "destination": {
                                "type": "string",
                                "description": "The destination of the message. must be 'user'"
                            }
                        },
                        "required": ["history"],
                        "additionalProperties": False
                    }
                }
            },
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "Ask coder agent to write basic code",
            #         "description": "This tool will ask our coder agent to write a basic code for you. You can ask for any code related query or task. It is an expert bioinformatician",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {

            #             },
            #             "required": ["history"],
            #             "additionalProperties": False
            #         }
            #     }
            # },
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "create_sequential_plan",
            #         "description": "Creates a detailed, step-by-step plan for a complex task or larger code-related query. Use this function for planning out intricate workflows or comprehensive code architectures. This function should also be used if asked for regenrate or modify the plan okay. ",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #             },
            #             "required": ["history"],
            #             "additionalProperties": False
            #         }
            #     }
            # },
        ]

        self.function_map = {
            "handle_simple_interaction": self.handle_simple_interaction,
            "write_basic_code": self.write_basic_code,
            "create_sequential_plan": self.create_sequential_plan,
        }


    def generate(self) -> tuple[str, "Generator"] | dict:
        """
        The Generate function is the standard agent function
        @return: Tuple of the system of message destination and the generator function to call
        """
        try:
            self.history.remove_system_messages()
            self.history.upsert(self.system_prompt)
            # Call the chat completion API
            response = chat_completion_function(self.history, tools=self.functions)

            if isinstance(response, openai.types.chat.ChatCompletion):
                message = response.choices[0].message
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name
                    function_arguments = tool_call.function.arguments
                    args_dict = json.loads(function_arguments) if function_arguments else {}
                    if function_name in self.function_map:
                        # Get the destination and response generator
                        destination, response_generator = self.function_map[function_name](**args_dict)
                        return destination, response_generator
                    else:
                        return "user", self.handle_simple_interaction(**args_dict) #type: ignore
                else:
                    return "user", self.handle_simple_interaction(**args_dict) #type: ignore
            else:
                return "user", self.handle_simple_interaction(**args_dict) #type: ignore
        except Exception as e:
            return "error",str(e) #type: ignore

    def route_message(self, destination: str, content: str) -> tuple[str, "Generator"]:
        # Separate code from text
        text, code = extract_code_and_text(content)
        # Prepare the response generator
        def response_generator():
            if text:
                yield {"type": "text", "content": text}
            if code:
                yield {"type": "code", "content": code}
        return destination, response_generator()

    def handle_simple_interaction(self, text, destination='user'):
        """
        Handles simple user interactions, separating text and code.

        Yields:
            Dict[str, str]: A dictionary with 'type' (either 'text' or 'code') and 'content'.
        """
        interaction_response = chat_completion_api(self.history, self.system_prompt)
        result = ""
        content_accumulated = ""

        def extract_code_and_text(content: str) -> dict[str, str]:
            """Helper function to split code and text."""
            code_blocks = re.findall(r'```(.*?)```', content, re.DOTALL)
            text_parts = re.split(r'```.*?```', content, flags=re.DOTALL)
            text = ' '.join([part.strip() for part in text_parts if part.strip()])
            code = '\n'.join([block.strip() for block in code_blocks if block.strip()])
            return {"text": text, "code": code}

        for chunk in interaction_response:
            try:
                # Accumulate content
                content = chunk['choices'][0]['delta']['content']
                if content:
                    content_accumulated += content
            except KeyError:
                continue

        # Once the response is fully accumulated, process it
        split_content = extract_code_and_text(content_accumulated)
        print("Split Content: ", split_content)
        # Yield text part
        if split_content['text']:
            print("Yielding Text: ", split_content['text'])
            yield {"type": "text", "content": split_content['text']}
            print("Yielded Text: ", split_content['text'])

        # Yield code part
        if split_content['code']:
            yield {"type": "code", "content": split_content['code']}

        # Log the full accumulated response
        self.history.log("assistant", content_accumulated)

    def write_basic_code(self):
        yield "Response: code"
        result = ""
        for chunk in chat_completion_api(self.history, system_prompt):
            try:
                content = chunk['choices'][0]['delta']['content']
                if content:
                    result += content
                yield chunk
            except:
                continue
        self.history.log("assistant", result)

    def decompose_complicated_task(self):
        decomposition_prompt = (
            f"You are an expert in task decomposition, particularly in bioinformatics and machine learning. "
            f"The user has provided a complex query that needs to be broken down into smaller, manageable subtasks. "
            f"Here is the query:\n\n"
            f"Query: '{self.history.most_recent_entry()}'\n\n"
            "Please identify the key components of this task and provide a step-by-step breakdown into smaller subtasks."
        )
        self.history.remove_system_messages()
        self.history.log("user", decomposition_prompt)
        yield "Response: task"
        result = ""
        for chunk in chat_completion_api(self.history, system_prompt):
            try:
                content = chunk['choices'][0]['delta']['content']
                if content:
                    result += content
                yield chunk
            except:
                continue
        self.history.log("assistant", result)

    def create_sequential_plan(self):
        self.history.remove_system_messages()
        yield "Response: plan"
        result = ""
        for chunk in chat_completion_plan(self.history, system_prompt):
            try:
                content = chunk['choices'][0]['delta']['content']
                if content:
                    result += content
                yield chunk
            except:
                continue
        self.history.log("assistant", result)
