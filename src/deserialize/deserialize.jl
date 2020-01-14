

struct CompGraphBuilder
   g::ONNX.Types.Graph
   sizes::Dict{String, <:Tuple}
   allnodes::Dict{String, ONNX.Types.Node}
   created::IdDict{ONNX.Types.Node, AbstractVertex}
   inputs::Vector{AbstractVertex}
end
CompGraphBuilder(g::ONNX.Types.Graph, sizes::Dict{String, <:Tuple}) = CompGraphBuilder(g, sizes, output_to_node(g.node), IdDict{ONNX.Types.Node, AbstractVertex}(), AbstractVertex[])
Broadcast.broadcastable(gb::CompGraphBuilder) = Ref(gb)
Broadcast.broadcastable(n::ONNX.Types.Node) = Ref(n)
function output_to_node(nodes)
   allnodes = Dict{String, ONNX.Types.Node}()
   for node in nodes
      for out in outputs(node)
         allnodes[out] = node
      end
   end
   return allnodes
end

node(name::String, gb::CompGraphBuilder, parent=nothing) = get!(gb.allnodes, name) do
    ONNX.Types.Node(String[], [parent.name], name, "Input", "", Dict{Any,Any}(:size => gb.sizes[name]), "")
end
nodes(gb::CompGraphBuilder) = keys(gb.allnodes)
innames(n::ONNX.Types.Node, gb::CompGraphBuilder) = innames(Val(optype(n)), n, gb)
innames(::Val, n::ONNX.Types.Node, gb::CompGraphBuilder) = inputs(n)[1:1]
innames(::Val{:Add}, n::ONNX.Types.Node, gb::CompGraphBuilder) = inputs(n)
innames(::Val{:Input}, ::ONNX.Types.Node, ::CompGraphBuilder) = []

innodes(n::ONNX.Types.Node, gb::CompGraphBuilder) = node.(innames(n, gb), gb, n)

Flux.params(n::ONNX.Types.Node, gb::CompGraphBuilder) = params(Val(optype(n)), n, gb)
Flux.params(::Val, n::ONNX.Types.Node, gb::CompGraphBuilder) = map(pname -> gb.g.initializer[pname], inputs(n)[2:end])
Flux.params(::Val{:Add}, n::ONNX.Types.Node, gb::CompGraphBuilder) = []

function extract(modelfile)
   f = readproto(open(modelfile), ONNX.Proto.ModelProto())
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


NaiveNASlib.CompGraph(filename::String) = CompGraph(extract(filename)...)
NaiveNASlib.CompGraph(m::ONNX.Types.Model, sizes) = CompGraph(m.graph, sizes)

function NaiveNASlib.CompGraph(g::ONNX.Types.Graph, sizes)
   gb = CompGraphBuilder(g, sizes)
   outputs::Vector{AbstractVertex} = vertex.(gb, node.(name.(g.output), gb))
   return CompGraph(gb.inputs, outputs)
end

NaiveNASlib.name(vi::ONNX.Types.ValueInfo) = vi.name
#name(n::ONNX.Types.Node) = n.name
NaiveNASlib.inputs(n::ONNX.Types.Node) = n.input
NaiveNASlib.outputs(n::ONNX.Types.Node) = n.output
optype(n::ONNX.Types.Node) = Symbol(n.op_type)


function vertex(gb::CompGraphBuilder, n::ONNX.Types.Node)::AbstractVertex
      return get!(gb.created, n) do
         n_create, ins = check_combine(gb, n)
         invertices = map(ni -> vertex(gb, ni), ins)
         v = create_vertex_default(gb, n_create, invertices)
         if isempty(nin(v))
            push!(gb.inputs, v)
         end
         return v
      end
end

function check_combine(gb::CompGraphBuilder, n::ONNX.Types.Node)
   ins = innodes(n, gb)
   # Case 1: Activation functions
   if optype(n) in keys(actfuns)
      if length(ins) == 1 && optype(ins[1]) in keys(actlayers)
         ins[1].attribute[:activation] = actfuns[optype(n)](n.attribute)
         return ins[1], innodes(ins[1], gb)
      end
   end

   # Case 2: Global pooling followed by reshape
   if optype(n) == :GlobalAveragePool
      if length(ins) == 1 && optype(ins[1]) == :Reshape
         ins[1].attribute[:wrap] = invariantops[optype(n)](n.attribute)
         return ins[1], innodes(ins[1], gb)
      end
   end

   return n, ins
end

create_vertex_default(gb::CompGraphBuilder, n::ONNX.Types.Node, inputs::Array{<:AbstractVertex}) = verts[optype(n)](n.name, inputs, n.attribute, params(n, gb)...)
