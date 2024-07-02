""" file taken from Orchid repo

Originally called orchid.jl in bin"""

#!/usr/bin/env -S julia -O3 --threads=auto --check-bounds=no

using Orchid
using SparseArrays
using LinearAlgebra
using Glob
using JSON
using ArgParse
using CodecZlib: GzipCompressor, GzipDecompressorStream

parse_edgelist(fp) =
    [parse.(Int, split(r)) for r in readlines(fp) for s in split(r) if s != ""]

"""
The Julia function parse_edgelist_collection(fp) reads an
edge list from a file fp (file path) and parses it into two separate collections: 
a vector of hypergraph labels y and a vector of vectors representing hyperedges rc. 
Eg (in the ih format: individual)
1 2
2 3 4
means we have one hypergraph with two hyper-edges: {1,2} and {2,3,4}. 
"""
function parse_edgelist_collection(fp)
    # Initialize variable
    rc, y = Vector{Int}[], Int[]
    # readlines(fp) reads all lines from the file specified by fp
    for r in readlines(fp)
        t = parse.(Int, split(r))
        # t[1] is the first element of the vector t, which is pushed onto the vector y.
        push!(y, t[1])
        # t[e:end] is the rest of the vector t, which is pushed onto the vector rc
        push!(rc, t[2:end])
    end
    y, rc
end

convert(m::AbstractSparseVector) = findnz(m)
convert(m::AbstractMatrix) = findnz(sparse(triu(m, 1)))
convert(m::Vector{<:Number}) = m
convert(m::Vector) = map(convert, m)
convert(s::Symbol) = String(s)
convert(s) = s

function get_entry(r, a, input, dispersion, alpha)
    (
        node_curvature_neighborhood = convert(r.node_curvature_neighborhood),
        directional_curvature = convert(r.directional_curvature),
        node_curvature_edges = convert(a.node_curvature_edges),
        edge_curvature = convert(a.edge_curvature),
        aggregation = convert(a.aggregation),
        dispersion = convert(dispersion),
        input = convert(input),
        alpha = convert(alpha),
    )
end

function run(input, dispersion, aggregation, alpha)
    !(0 <= alpha <= 1) && throw("!(0 <= alpha <= 1)")

    # The measure. This is just picking wich measure we use.
    D = Dict{String,Type}(
        lowercase("UnweightedClique") => Orchid.DisperseUnweightedClique,
        lowercase("WeightedClique") => Orchid.DisperseWeightedClique,
        lowercase("UnweightedStar") => Orchid.DisperseUnweightedStar,
    )[lowercase(dispersion)]
    # The aggregation function. This is literally picking the agregation function
    A = Dict{String,Any}(
        "mean" => Orchid.AggregateMean,
        "max" => Orchid.AggregateMax,
        "all" => (Orchid.AggregateMean, Orchid.AggregateMax),
    )[lowercase(aggregation)]

    # @info "A is $A"

    # @info "D is $D"

    guess_cost_calc(E) =
        length(E) > 10_000 || maximum(e -> maximum(e; init = 0), E) > 10_000 ?
        Orchid.CostOndemand : Orchid.CostMatrix
    open_() = endswith(input, ".gz") ? GzipDecompressorStream(open(input)) : open(input)

    if occursin(".chg.tsv", input) # collection of hypergraphs
        @info "Reading Hypergraphs"
        y, rc = parse_edgelist_collection(open_())
        # y is the hypergraph number this edge belongs to
        # rc is a list of list. Each element is an hyperedge, with the the vertex in that hyperedge
        # @info "y is $y"
        # @info "rc is $rc"
        ys = unique(y) # these are the labels of the hypegraphs. eg if we have two and they are 
        # labelled 2 and 0, this would be [2,0]
        # @info "the ys are $ys"
        Tot = length(ys) # this is the total number of hypergraphs in the file 
        # @info "Tot is $Tot"
        results = []
        # loop through the hypergraphs (the ys)
        foreach(ys) do Y
            @info "Importing Hypergraph $Y/$Tot"
            @info "Y is $Y" # this is the hypergraph number
            # so here they filter for the hyperedges in hypergraph number Y
            E = rc[y.==Y] # set of hyper-edges 
            r = Orchid.hypergraph_curvatures(D, A, E, alpha, guess_cost_calc(E))
            for a in r.aggregations
                push!(results, get_entry(r, a, input, dispersion, alpha))
            end
        end
        results
    else # individual hypergraph
        @info "Importing Hypergraph"
        E = parse_edgelist(open_())
        # @info "A is $A"
        # A is the agrregation function (Eg aggregate mean, aggregate max)
        # @info "D is $D"
        # I think D is the measure
        @info "E is $E" #
        r = Orchid.hypergraph_curvatures(D, A, E, alpha, guess_cost_calc(E))
        map(r.aggregations) do a
            get_entry(r, a, input, dispersion, alpha)
        end
    end
end

function orchid_main(
    input::String,
    output::String = "-";
    dispersion::String = "UnweightedClique",
    aggregation::String = "All",
    alpha = 0.1,
)
    if !occursin("*", input)
        results = run(input, dispersion, aggregation, alpha)
        @info "Converting Curvatures to JSON"
        j = JSON.json(results)
        @info "Writing JSON to $output"
        write(
            output == "-" ? stdout : open(output, "w"),
            endswith(output, "gz") ? transcode(GzipCompressor, j) : j * "\n",
        )
    else
        @info "Globbing $input"
        results =
            [a for input in glob(input) for a in run(input, dispersion, aggregation, alpha)]
        @info "Converting Curvatures to JSON"
        j = JSON.json(results)
        @info "Writing JSON to $output"
        write(
            output == "-" ? stdout : open(output, "w"),
            endswith(output, "gz") ? transcode(GzipCompressor, j) : j * "\n",
        )
    end
end

function main()
    s = ArgParseSettings(
        description = """
    This is a command line interface for the ORCHID hypergraph curvature framework described in 

    Calls orchid_main
    
    Coupette, C., Dalleiger, S. and Rieck, B., 
    Ollivier-Ricci Curvature for Hypergraphs: A Unified Framework, 
    ICLR 2023, doi:10.48550/arXiv.2210.12048.
""",
    )
    @add_arg_table! s begin
        "-i", "--input"
        required = true
        help = "Input hypergraph(s) in edgelist format (options: individual edgelist for one hypergraph, collection of edgelists [ext: chg.tsv[.gz]], or a globbing pattern ['*' in `input`] both for multiple hypergraphs)"
        "-o", "--output"
        required = false
        default = "-"
        help = "Output destination ['-' denotes stdout]"
        "--dispersion"
        default = "UnweightedClique"
        help = "Dispersion (options: UnweightedClique, WeightedClique, or UnweightedStar)"
        "--aggregation"
        default = "Mean"
        help = "Aggregation (options: Mean, Max, or All)"
        "--alpha"
        arg_type = Float64
        help = "Self-Dispersion Weight"
        default = 0.0
    end
    opts = parse_args(s)
    @info "Hello. We are computing ORC. You care calling with options $opts['input'], $opts['output']"

    orchid_main(
        opts["input"],
        opts["output"];
        dispersion = opts["dispersion"],
        aggregation = opts["aggregation"],
        alpha = opts["alpha"],
    )
end

main()
