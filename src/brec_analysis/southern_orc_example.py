
from scipy.stats import ks_2samp

from encodings_hnns.orc_from_southern import (
    ollivier_ricci_curvature,
    prob_rw,
    prob_two_hop,
)


def southern_orc_example(rook, shrikhande) -> None:
    # After loading rook and shrikhande graphs
    print("Computing ORCs for Rook and Shrikhande graphs...")

    # Compute ORCs with different probability measures and alpha values
    alpha_values = [0.0, 0.5]
    prob_measures = {"Default": None, "Random Walk": prob_rw, "Two Hop": prob_two_hop}

    for alpha in alpha_values:
        print(f"\nResults with alpha = {alpha}:")
        for measure_name, prob_fn in prob_measures.items():
            print(f"\n{measure_name} probability measure:")

            rook_orc = ollivier_ricci_curvature(rook, alpha=alpha, prob_fn=prob_fn)
            shrikhande_orc = ollivier_ricci_curvature(
                shrikhande, alpha=alpha, prob_fn=prob_fn
            )

            print("Rook Graph:")
            print(f"Mean curvature: {rook_orc.mean():.4f}")
            print(f"Min curvature: {rook_orc.min():.4f}")
            print(f"Max curvature: {rook_orc.max():.4f}")

            print("\nShrikhande Graph:")
            print(f"Mean curvature: {shrikhande_orc.mean():.4f}")
            print(f"Min curvature: {shrikhande_orc.min():.4f}")
            print(f"Max curvature: {shrikhande_orc.max():.4f}")

            # Check if distributions are different
            stat, pval = ks_2samp(rook_orc, shrikhande_orc)
            print(f"\nKS test p-value: {pval:.4f}")
            print(f"Distributions are {'different' if pval < 0.05 else 'similar'}")
