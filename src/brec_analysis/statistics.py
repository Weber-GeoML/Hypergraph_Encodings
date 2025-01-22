"""Functions for calculating statistics and generating LaTeX tables."""

from brec_analysis.match_status import MatchStatus
from typing import Dict

def calculate_statistics(
    results_by_category: dict, 
    consider_scaling: bool = True
) -> Dict[str, Dict[str, float]]:
    """Calculate percentage of different encodings for each category.
    
    Args:
        results_by_category: Dictionary with results by category
        consider_scaling: If True, count SCALED_MATCH as a match
        
    Returns:
        Dictionary with statistics by category and encoding
    """
    stats = {}
    for category in results_by_category:
        stats[category] = {}
        for level in ["graph_level", "hypergraph_level"]:
            for encoding in results_by_category[category][level]:
                results = results_by_category[category][level][encoding]
                
                if consider_scaling:
                    # Count as different only if NO_MATCH
                    different = sum(1 for r in results if r == MatchStatus.NO_MATCH)
                else:
                    # Count as different if not EXACT_MATCH
                    different = sum(1 for r in results 
                                  if r != MatchStatus.EXACT_MATCH)
                
                total = len(results)
                if total > 0:  # Avoid division by zero
                    percentage = (different / total) * 100
                    key = f"{level}_{encoding}"
                    stats[category][key] = percentage
                
    return stats

def generate_latex_table(stats: dict) -> str:
    """Generate LaTeX table from statistics."""
    #TODO
    pass
