#!/usr/bin/env python3
import json
import gzip
import argparse
import glob
from orchid import DisperseUnweightedClique, DisperseWeightedClique, DisperseUnweightedStar, AggregateMean, AggregateMax, hypergraph_curvatures, CostOndemand, CostMatrix

def parse_edgelist(fp):
    """Parse an edge list file into a list of lists of integers."""
    return [list(map(int, line.split())) for line in open(fp) if line.strip()]

def parse_edgelist_collection(fp):
    """Parse a collection of edge lists from a file into two lists: y and rc."""
    y = []
    rc = []
    for line in open(fp):
        t = list(map(int, line.split()))
        y.append(t[0])
        rc.append(t[1:])
    return y, rc

def get_entry(r, a, input_file, dispersion, alpha):
    """Create a dictionary entry from hypergraph curvature results."""
    return {
        'node_curvature_neighborhood': list(r.node_curvature_neighborhood),
        'directional_curvature': list(r.directional_curvature),
        'node_curvature_edges': list(a.node_curvature_edges),
        'edge_curvature': list(a.edge_curvature),
        'aggregation': list(a.aggregation),
        'dispersion': dispersion,
        'input': input_file,
        'alpha': alpha
    }

def run(input_file, dispersion, aggregation, alpha):
    """Run the ORCHID hypergraph curvature analysis."""
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    # Define mappings for dispersion and aggregation types
    D = {
        'unweightedclique': DisperseUnweightedClique,
        'weightedclique': DisperseWeightedClique,
        'unweightedstar': DisperseUnweightedStar
    }[dispersion.lower()]

    A = {
        'mean': AggregateMean,
        'max': AggregateMax,
        'all': (AggregateMean, AggregateMax)
    }[aggregation.lower()]

    # Function to guess cost calculation method based on edge count
    def guess_cost_calc(E):
        if len(E) > 10000 or max(max(e, default=0) for e in E) > 10000:
            return CostOndemand
        else:
            return CostMatrix

    # Function to open files, supporting gzip compression
    def open_file():
        if input_file.endswith(".gz"):
            return gzip.open(input_file, 'rt')
        else:
            return open(input_file, 'r')

    if ".chg.tsv" in input_file:
        print("Reading Hypergraphs")
        y, rc = parse_edgelist_collection(open_file())
        ys = set(y)
        Tot = len(ys)
        results = []
        for Y in ys:
            print(f"Importing Hypergraph {Y}/{Tot}")
            E = [rc[i] for i in range(len(y)) if y[i] == Y]
            r = hypergraph_curvatures(D, A, E, alpha, guess_cost_calc(E))
            for a in r.aggregations:
                results.append(get_entry(r, a, input_file, dispersion, alpha))
        return results
    else:
        print("Importing Hypergraph")
        E = parse_edgelist(open_file())
        r = hypergraph_curvatures(D, A, E, alpha, guess_cost_calc(E))
        return [get_entry(r, a, input_file, dispersion, alpha) for a in r.aggregations]

def orchid_main(input_file, output="-", dispersion="UnweightedClique", aggregation="All", alpha=0.1):
    """Main function to run ORCHID analysis and output results as JSON."""
    if '*' not in input_file:
        results = run(input_file, dispersion, aggregation, alpha)
        print("Converting Curvatures to JSON")
        j = json.dumps(results)
        print(f"Writing JSON to {output}")
        with (gzip.open(output, 'wt') if output.endswith('.gz') else open(output, 'w')) as f:
            f.write(j + "\n")
    else:
        print(f"Globbing {input_file}")
        results = [a for infile in glob.glob(input_file) for a in run(infile, dispersion, aggregation, alpha)]
        print("Converting Curvatures to JSON")
        j = json.dumps(results)
        print(f"Writing JSON to {output}")
        with (gzip.open(output, 'wt') if output.endswith('.gz') else open(output, 'w')) as f:
            f.write(j + "\n")

################################
############ Main ##############
################################

def main():
    parser = argparse.ArgumentParser(description="""
        This is a command line interface for the ORCHID hypergraph curvature framework described in 
        
        Coupette, C., Dalleiger, S. and Rieck, B., 
        Ollivier-Ricci Curvature for Hypergraphs: A Unified Framework, 
        ICLR 2023, doi:10.48550/arXiv.2210.12048.
    """)
    parser.add_argument('-i', '--input', required=True, help="Input hypergraph(s) in edgelist format")
    parser.add_argument('-o', '--output', default="-", help="Output destination")
    parser.add_argument('--dispersion', default="UnweightedClique", help="Dispersion type")
    parser.add_argument('--aggregation', default="Mean", help="Aggregation type")
    parser.add_argument('--alpha', type=float, default=0.1, help="Self-Dispersion Weight")
    args = parser.parse_args()

    orchid_main(args.input, args.output, dispersion=args.dispersion, aggregation=args.aggregation, alpha=args.alpha)

if __name__ == "__main__":
    main()
