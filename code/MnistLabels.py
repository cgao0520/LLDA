import gzip
import numpy as np
from collections import Counter


def count_mnist_labels(labels_path):
    """
    Parses the MNIST idx1-ubyte file and returns the count of each digit.
    Handles both compressed (.gz) and uncompressed files.
    """
    # Use gzip if the file ends in .gz, otherwise open normally
    opener = gzip.open if labels_path.endswith('.gz') else open
    
    with opener(labels_path, 'rb') as f:
        # Read the header: Magic Number (4 bytes) and Item Count (4 bytes)
        # '>II' means Big-Endian, two Unsigned Integers
        magic, count = np.frombuffer(f.read(8), dtype='>i4')
        
        # Load all labels into a numpy array (starting after the 8-byte header)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    # Count occurrences using numpy's unique function
    digits, counts = np.unique(labels, return_counts=True)
    
    # Return a dictionary of {digit: count}
    return dict(zip(digits, counts))
