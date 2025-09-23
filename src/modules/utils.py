import pickle
import pandas as pd
from typing import Any

def save_object(obj: Any, path: str) -> None:
    """
    Save a Python object to a file using pickle serialization.

    Args:
        obj (Any): The Python object to serialize and save.
        path (str): The file path where the object will be stored.
    """
    # Open the file in binary write mode
    with open(path, 'wb') as f:
        # Serialize and write the object to the file
        pickle.dump(obj, f)

    return

def load_object(path: str) -> Any:
    """
    Load a Python object from a pickle file.

    Args:
        path (str): The file path from which to load the object.

    Returns:
        Any: The deserialized Python object.
    """
    # Open the file in binary read mode
    with open(path, 'rb') as f:
        # Load and return the serialized object
        obj = pickle.load(f)

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
    df = pd.read_csv(path, compression=compression)

    return df
