import matplotlib.pyplot as plt
import numpy as np

def plot_face_similarity(embedding1, embedding2, output_path=None):
    """
    Plots a comparison of two face embeddings.

    Parameters:
    - embedding1: numpy array of face embedding (uploaded face)
    - embedding2: numpy array of face embedding (matched face)
    - output_path: optional path to save the plot
    """

    if len(embedding1) != len(embedding2):
        raise ValueError("Embeddings must be of the same length")

    indices = np.arange(len(embedding1))

    plt.figure(figsize=(14, 5))
    plt.plot(indices, embedding1, label='Uploaded Face', marker='o')
    plt.plot(indices, embedding2, label='Matched Face', marker='x')
    plt.fill_between(indices, embedding1, embedding2, color='gray', alpha=0.2)

    plt.title("Face Embedding Similarity")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
        print(f"Similarity plot saved to {output_path}")
    else:
        plt.show()
