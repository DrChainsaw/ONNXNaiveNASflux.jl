
"""
    SourceVertex{T} <: AbstractVertex
    SourceVertex(data::T, name::String)

`AbstractVertex` which returns `data` when invoked with no arguments.
"""
struct SourceVertex{T} <: AbstractVertex
    data::T
    name::String
end
SourceVertex(data) = SourceVertex(data, "SourceVertex")

(v::SourceVertex)() = v.data

NaiveNASlib.name(v::SourceVertex) = v.name
NaiveNASlib.trait(v::SourceVertex) = Immutable()

NaiveNASflux.layertype(v::SourceVertex) = guess_layertype(ndims(v.data))

NaiveNASlib.nout(v::SourceVertex) = size(v.data, NaiveNASflux.outdim(layertype(v)))
NaiveNASlib.nin(::SourceVertex) = Integer[]

# TODO: Move to NaiveNASlib
NaiveNASlib.nin(v::OutputsVertex) = nin(base(v))
NaiveNASlib.nout(v::OutputsVertex) = nout(base(v))

NaiveNASlib.inputs(::SourceVertex) = AbstractVertex[]

function sourcevertex_with_outputs(data, name)
    sv = SourceVertex(data, name)
    ov = OutputsVertex(sv)
    NaiveNASlib.init!(ov, sv)
    return ov
end
