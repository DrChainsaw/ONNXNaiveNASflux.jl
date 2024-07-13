

"""
   extract(modelfile)

Return a [`ONNX.Types.Model`](@ref) and a Dict mapping input variables to size tuples (in Flux order).

Beware that missing/variable size data for a dimension results in a random size for that dimension. Therefore sizes should mostly be used to determine the number of dimensions.
"""
extract(modelfile::AbstractString) = open(io -> extract(io), modelfile)
extract(io::IO) = ONNX.readproto(io, ONNX.ModelProto)

NaiveNASlib.name(n::OnnxNode) = name(n.proto)
NaiveNASlib.name(n::ONNX.NodeProto) = n.name
NaiveNASlib.name(vip::ONNX.ValueInfoProto) = vip.name
NaiveNASlib.name(tp::ONNX.TensorProto) = tp.name

load(filename::String, insizes...; kwargs...) = open(io -> load(io, insizes...; kwargs...), filename)
load(io::IO, insizes...; kwargs...) = load(extract(io), insizes...; kwargs...)
load(m::ONNX.ModelProto, insizes...; kwargs...) = load(m.graph, insizes...; kwargs...)
load(g::ONNX.GraphProto, insizes...; kwargs...) = CompGraph(g, insizes...; kwargs...)
NaiveNASlib.CompGraph(g::ONNX.GraphProto, insizes...; kwargs...) = CompGraph(CompGraphBuilder(g, insizes...); kwargs...)
function NaiveNASlib.CompGraph(gb::CompGraphBuilder; vfun = create_vertex_default, infer_shapes=true)
    outputs::Vector{AbstractVertex} = vertex.(gb, node.(name.(gb.g.output), gb), vfun)
    graph = CompGraph(gb.inputs, outputs)
    if infer_shapes
        try_infer_sizes!(graph, (get(gb.sizes, n, (missing,)) for n in name.(inputs(graph)))...)
    end
    return graph
end

"""
   vertex(gb::CompGraphBuilder, n::ONNX.Types.Node, vfun = create_vertex_default)

Return an `AbstractVertex` created from `n`.

Inputs to the returned vertex are created recursively based on state in `gb`.
"""
function vertex(gb::CompGraphBuilder, n::OnnxNode, vfun = create_vertex_default)
      return get!(gb.created, n) do
         n_create, ins = check_combine(gb, n)
         invertices = map(ni -> vertex(gb, ni, vfun), ins)
         v = vfun(n_create, invertices)
         if is_input(v)
            push!(gb.inputs, v)
         end
         return v
      end
end

is_input(v::AbstractVertex) = is_input(base(v))
is_input(v::InputVertex) = true
is_input(v::CompVertex) = false
is_input(v::SourceVertex) = false


create_vertex_default(n::OnnxNode, inputs::Array; kwargs...) = verts[optype(n)](name(n), inputs, n.attribute, params(n)...; kwargs...)
