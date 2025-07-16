import scanpy as sc

def load_adata(file_path):
    """
    Loads an AnnData object from the specified file path.
    
    Parameters:
    - file_path (str): Path to the AnnData file.
    
    Returns:
    - adata (AnnData): Loaded AnnData object.
    """
    try:
        adata = sc.read(file_path)
        print(f"Successfully loaded AnnData object from {file_path}")
        return adata
    except Exception as e:
        print(f"Error loading AnnData object: {e}")
        return None