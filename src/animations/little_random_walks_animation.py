from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def create_random_walk_gif():
    """Create a gif of a random walk on a graph."""
    # Create output directory
    output_dir = Path("random_walk_frames")
    output_dir.mkdir(exist_ok=True)

    # Get Karate Club graph and set fixed layout
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G, seed=42)  # Fixed layout with seed

    # Parameters
    start_node = 0  # Starting node
    n_steps = 31  # Number of steps in random walk
    current_node = start_node

    # Perform random walk and create frames
    frames = []
    for step in range(n_steps):
        plt.figure(figsize=(8, 8))

        # Draw all nodes in default color
        nx.draw(
            G,
            pos,
            node_color="lightblue",
            node_size=500,
            with_labels=True,
            font_weight="bold",
        )

        # Draw current node in red
        nx.draw_networkx_nodes(
            G, pos, nodelist=[current_node], node_color="red", node_size=500
        )

        plt.title(f"Random Walk Step {step}", fontsize=16, pad=20)

        # Save frame
        frame_path = output_dir / f"frame_{step:03d}.png"
        plt.savefig(frame_path, bbox_inches="tight")
        frames.append(frame_path)
        plt.close()

        # Get next node
        neighbors = list(G.neighbors(current_node))
        current_node = np.random.choice(neighbors)

    # Create GIF
    images = [imageio.imread(str(frame)) for frame in frames]
    imageio.mimsave(
        "gifs/random_walk.gif", images, duration=60
    )  # Half second per frame

    # Clean up frame files
    for frame in frames:
        frame.unlink()
    output_dir.rmdir()


if __name__ == "__main__":
    create_random_walk_gif()
