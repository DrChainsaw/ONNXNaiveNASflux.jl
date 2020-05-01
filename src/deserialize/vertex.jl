
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

NaiveNASlib.clone(v::SourceVertex;cf=clone) = SourceVertex(cf(v.data, cf=cf), cf(v.name,cf=cf))

NaiveNASlib.name(v::SourceVertex) = v.name
NaiveNASlib.trait(v::SourceVertex) = Immutable()

NaiveNASflux.layertype(v::SourceVertex) = guess_layertype(ndims(v.data))

NaiveNASlib.nout(v::SourceVertex) = size(v.data, NaiveNASflux.outdim(layertype(v)))
NaiveNASlib.nin(::SourceVertex) = Integer[]

NaiveNASlib.nout_org(v::SourceVertex) = nout(v)
NaiveNASlib.nin_org(v::SourceVertex) = nin(v)

# TODO: Move to NaiveNASlib
NaiveNASlib.nin(v::OutputsVertex) = nin(base(v))
NaiveNASlib.nin_org(v::OutputsVertex) = nin_org(base(v))
NaiveNASlib.nout(v::OutputsVertex) = nout(base(v))
NaiveNASlib.nout_org(v::OutputsVertex) = nout_org(base(v))

NaiveNASlib.inputs(::SourceVertex) = AbstractVertex[]

NaiveNASlib.minΔninfactor_only_for(v::SourceVertex,s=[]) = missing
NaiveNASlib.minΔnoutfactor_only_for(v::SourceVertex,s=[]) = missing

function sourcevertex_with_outputs(data, name)
    sv = SourceVertex(data, name)
    ov = OutputsVertex(sv)
    NaiveNASlib.init!(ov, sv)
    return ov
end
