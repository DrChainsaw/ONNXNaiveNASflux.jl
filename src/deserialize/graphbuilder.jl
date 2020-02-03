

"""
   CompGraphBuilder(g::ONNX.Types.Graph, sizes::Dict{String, <:Tuple})

Helper struct for holding state when constructing a CompGraph out of the provided [`ONNX.Types.Graph`](@ref).
"""
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
innames(::Val{:Input}, ::ONNX.Types.Node, ::CompGraphBuilder) = []
innames(::Val{:Add}, n::ONNX.Types.Node, ::CompGraphBuilder) = inputs(n)
innames(::Val{:Concat}, n::ONNX.Types.Node, ::CompGraphBuilder) = inputs(n)

innodes(n::ONNX.Types.Node, gb::CompGraphBuilder) = node.(innames(n, gb), gb, n)

Flux.params(n::ONNX.Types.Node, gb::CompGraphBuilder) = params(Val(optype(n)), n, gb)
Flux.params(::Val, n::ONNX.Types.Node, gb::CompGraphBuilder) = map(pname -> gb.g.initializer[pname], setdiff(inputs(n), innames(n, gb)))
