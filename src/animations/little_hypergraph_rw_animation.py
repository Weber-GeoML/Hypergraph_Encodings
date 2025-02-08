from pathlib import Path

import hypernetx as hnx
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from encodings_hnns.laplacians import Laplacians


def create_hypergraph_random_walk_gif():
    """Create a gif of a random walk on a hypergraph."""
    # Create output directory
    output_dir = Path("hypergraph_random_walk_frames")
    output_dir.mkdir(exist_ok=True)

    # Create a simple example hypergraph
    hypergraph = {
        "e1": [0, 1, 2],  # A 3-node hyperedge
        "e2": [2, 3, 4],  # Another 3-node hyperedge
        "e3": [4, 5],  # A 2-node hyperedge
        "e4": [0, 5, 6],  # A 3-node hyperedge
    }

    # Create dataset dictionary in the required format
    dataset = {
        "hypergraph": hypergraph,
        "n": 7,  # Number of nodes (0-6)
        "features": [[1.0] for _ in range(7)],  # Simple 1D features
        "labels": np.array([[1] if i % 2 == 0 else [0] for i in range(7)]),
    }

    # Initialize Laplacian for random walk
    laplacian = Laplacians(dataset)
    laplacian.compute_random_walk_laplacian(
        rw_type="EN"
    )  # Using equal node random walk

    # Get transition matrix
    rw_matrix = -laplacian.rw_laplacian + np.eye(dataset["n"])

    # Parameters
    start_node = 0
    n_steps = 101
    current_node = start_node

    # Create HyperNetX hypergraph
    H = hnx.Hypergraph(hypergraph)
    pos = nx.spring_layout(H.bipartite())

    # Perform random walk and create frames
    frames = []
    for step in range(n_steps):
        plt.figure(figsize=(10, 10))

        # Draw hypergraph with all nodes in default color
        hnx.draw(
            H,
            pos=pos,
            with_node_labels=True,
            with_edge_labels=True,
        )

        # Highlight current node in red
        plt.scatter(*pos[current_node], color="red", s=500, zorder=10)

        plt.title(f"Hypergraph Random Walk Step {step}", fontsize=16, pad=20)

        # Save frame
        frame_path = output_dir / f"frame_{step:03d}.png"
        plt.savefig(frame_path, bbox_inches="tight")
        frames.append(frame_path)
        plt.close()

        # Get next node using transition probabilities
        probs = rw_matrix[current_node]
        next_node = np.random.choice(range(dataset["n"]), p=probs)
        current_node = next_node

    # Create GIF
    images = [imageio.imread(str(frame)) for frame in frames]
    imageio.mimsave("gifs/hypergraph_random_walk.gif", images, duration=50.5)
    print("saved to hypergraph_random_walk.gif")

    # Clean up frame files
    for frame in frames:
        frame.unlink()
    output_dir.rmdir()


if __name__ == "__main__":
    create_hypergraph_random_walk_gif()
