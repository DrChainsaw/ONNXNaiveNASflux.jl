

"""
   extract(modelfile)

Return a [`ONNX.Types.Model`](@ref) and a Dict mapping input variables to size tuples (in Flux order).

Beware that missing/variable size data for a dimension results in a random size for that dimension. Therefore sizes should mostly be used to determine the number of dimensions.
"""
extract(modelfile::AbstractString) = open(io -> extract(io), modelfile)
function extract(io::IO)
   f = readproto(io, ONNX.Proto.ModelProto())
   return convert(f), sizes(f)
end

sizes(mp::ONNX.Proto.ModelProto) = sizes(mp.graph)
sizes(gp::ONNX.Proto.GraphProto) = Dict(name.(gp.input) .=> size.(gp.input))

NaiveNASlib.name(vip::ONNX.Proto.ValueInfoProto) = vip.name

Base.size(vip::ONNX.Proto.ValueInfoProto) = size(vip._type)
Base.size(tp::ONNX.Proto.TypeProto) = size(tp.tensor_type)
Base.size(tp_t::ONNX.Proto.TypeProto_Tensor) = size(tp_t.shape)
Base.size(tsp::ONNX.Proto.TensorShapeProto) = size.(Tuple(reverse(tsp.dim)))
Base.size(tsp_d::ONNX.Proto.TensorShapeProto_Dimension) = isdefined(tsp_d, :dim_value) ? tsp_d.dim_value : missing

"""
   CompGraph(filename::String)

Return a [`CompGraph`](@ref) loaded from the given file.
"""
NaiveNASlib.CompGraph(filename::String, vfun = create_vertex_default) = open(io -> CompGraph(io), filename)
NaiveNASlib.CompGraph(io::IO, vfun = create_vertex_default) = CompGraph(extract(io)..., vfun)
NaiveNASlib.CompGraph(m::ONNX.Types.Model, sizes, vfun = create_vertex_default) = CompGraph(m.graph, sizes, vfun)

function NaiveNASlib.CompGraph(g::ONNX.Types.Graph, sizes, vfun = create_vertex_default)
   gb = CompGraphBuilder(g, sizes)
   outputs::Vector{AbstractVertex} = vertex.(gb, node.(name.(g.output), gb), vfun)
   return CompGraph(gb.inputs, outputs)
end

NaiveNASlib.name(vi::ONNX.Types.ValueInfo) = vi.name
#name(n::ONNX.Types.Node) = n.name
NaiveNASlib.inputs(n::ONNX.Types.Node) = n.input
NaiveNASlib.outputs(n::ONNX.Types.Node) = n.output
optype(n::ONNX.Types.Node) = Symbol(n.op_type)

"""
   vertex(gb::CompGraphBuilder, n::ONNX.Types.Node)::AbstractVertex

Return an `AbstractVertex` created from `n`.

Inputs to the returned vertex are created recursively based on state in `gb`.
"""
function vertex(gb::CompGraphBuilder, n::ONNX.Types.Node, vfun = create_vertex_default)::AbstractVertex
      return get!(gb.created, n) do
         n_create, ins = check_combine(gb, n)
         invertices = map(ni -> vertex(gb, ni, vfun), ins)
         # TODO: Let user supply this...
         v = vfun(gb, n_create, invertices)
         if isempty(nin(v))
            push!(gb.inputs, v)
         end
         return v
      end
end

"""
   check_combine(gb::CompGraphBuilder, n::ONNX.Types.Node)

Return a potentially combined [`ONNX.Types.Node`](@ref) and its inputs.

Purpose is to handle e.g activation functions to layers which Flux has as member of the layer struct while ONNX has as a separate node.

Main motivation is an attempt to make serialization/deserialization "pure" in the sense that `g -> serialize -> deserialize === g`.

Another motivation is that mutation becomes a bit harder if things like global pooling and dropping of dimensions are separated into two vertices.
"""
function check_combine(gb::CompGraphBuilder, n::ONNX.Types.Node)
   ins = innodes(n, gb)
   # Case 1: Activation functions
   if optype(n) in keys(actfuns)
      if length(ins) == 1 && optype(ins[1]) in keys(actlayers)
         ins[1].attribute[:activation] = actfuns[optype(n)](n.attribute, params(n, gb)...)
         return ins[1], innodes(ins[1], gb)
      end
   end

   # Case 2: Global pooling followed by reshape
   if any(ot -> ot == optype(n), (:Reshape, :Squeeze))
      if length(ins) == 1 && optype(ins[1]) == :GlobalAveragePool
         ins[1].attribute[:wrap] = invariantops[optype(n)](n.attribute, params(n, gb)...)
         return ins[1], innodes(ins[1], gb)
      end
   end

   return n, ins
end

create_vertex_default(gb::CompGraphBuilder, n::ONNX.Types.Node, inputs::Array{<:AbstractVertex}; kwargs...) = verts[optype(n)](n.name, inputs, n.attribute, params(n, gb)...; kwargs...)
