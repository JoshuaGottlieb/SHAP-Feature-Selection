import bz2
import gzip
import lzma
import os
import pickle
from typing import Any, Optional

import pandas as pd

def save_object(obj: Any, path: str, compression: Optional[str] = None) -> None:
    """
    Serialize and save a Python object (e.g., model, transformer, dictionary) to disk,
    with optional compression.

    This function ensures that the destination directory exists before writing,
    supports multiple compression formats, and appends an appropriate file extension
    based on the compression type.

    Supported compression formats:
        - None: Uncompressed `.pickle`
        - 'gzip': GZIP-compressed `.pickle.gz`
        - 'bz2': BZ2-compressed `.pickle.bz2`
        - 'lzma': LZMA/XZ-compressed `.pickle.xz`

    Args:
        obj (Any):
            The Python object to serialize and save.
        path (str):
            Destination file path (without extension).
            Example: `"models/random_forest_model"`
        compression (str, optional):
            Compression type to use ('gzip', 'bz2', 'lzma', or None).
            Defaults to None (uncompressed).
    """
    # --- Step 1: Ensure the output directory exists ---
    root = os.path.dirname(path)
    if root and not os.path.exists(root):
        os.makedirs(root)

    # --- Step 2: Handle supported compression formats ---
    if compression in ["gzip", "bz2", "lzma"]:
        if compression == "gzip":
            ext = ".pickle.gz"
            with gzip.open(path + ext, "wb") as f:
                pickle.dump(obj, f)
        elif compression == "bz2":
            ext = ".pickle.bz2"
            with bz2.BZ2File(path + ext, "wb") as f:
                pickle.dump(obj, f)
        elif compression == "lzma":
            ext = ".pickle.xz"
            with lzma.open(path + ext, "wb") as f:
                pickle.dump(obj, f)

    else:
        # --- Step 3: Save as an uncompressed pickle file ---
        if compression is not None:
            print("Warning: Unknown compression type. Defaulting to uncompressed pickle format.")
        ext = ".pickle"
        with open(path + ext, "wb") as f:
            pickle.dump(obj, f)

    # --- Step 4: Print confirmation ---
    print(f"Successfully saved object to {path + ext}")

    return

def load_object(path: str) -> Any:
    """
    Load and deserialize a Python object (e.g., model, transformer, dictionary)
    from disk, automatically handling compressed pickle formats.

    This function supports the same compression extensions as `save_object()`:
        - `.pickle`: Uncompressed
        - `.pickle.gz`: GZIP-compressed
        - `.pickle.bz2`: BZ2-compressed
        - `.pickle.xz`: LZMA/XZ-compressed

    Args:
        path (str):
            Full path to the serialized object file, including its extension.
            Example: `"models/random_forest_model.pickle.gz"`

    Returns:
        Any:
            The deserialized Python object.
    """
    # --- Step 1: Determine compression type based on file extension ---
    if path.endswith(".pickle.gz"):
        compression = "gzip"
    elif path.endswith(".pickle.bz2"):
        compression = "bz2"
    elif path.endswith(".pickle.xz"):
        compression = "lzma"
    else:
        compression = None

    # --- Step 2: Load object using appropriate method ---
    if compression == "gzip":
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
    elif compression == "bz2":
        with bz2.BZ2File(path, "rb") as f:
            obj = pickle.load(f)
    elif compression == "lzma":
        with lzma.open(path, "rb") as f:
            obj = pickle.load(f)
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    # --- Step 3: Return the loaded Python object ---
    print(f"Successfully loaded object from {path}")
    
    return obj

def load_dataset(path: str, compression: str = 'gzip') -> pd.DataFrame:
    """
    Load a dataset from a CSV file into a pandas DataFrame.

    Args:
        path (str): The file path to the dataset.
        compression (str, optional): Compression type used on the CSV file 
            (e.g., 'gzip', 'bz2', 'zip', or None). Defaults to 'gzip'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.
    """
    # Read CSV file into a DataFrame with optional compression
    df = pd.read_csv(path, compression = compression)

    return df