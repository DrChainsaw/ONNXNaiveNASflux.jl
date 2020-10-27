
"""
   OnnxNode(proto::ONNX.NodeProto, params::Vector{ONNX.TensorProto}, attribs::Dict)

Helper struct for holding state for a NodeProto when constructing a CompGraph
"""
struct OnnxNode
   proto::ONNX.NodeProto
   params::Vector{ONNX.TensorProto}
   attribute::Dict{Symbol, Any} # Must be Any or else we might overspecialize, preventing that stuff is added later
end
OnnxNode(proto, ps) = OnnxNode(proto, ps, Dict{Symbol, Any}(Dict(proto.attribute)))

"""
   CompGraphBuilder(g::ONNX.Types.Graph, sizes::Dict{String, <:Tuple})

Helper struct for holding state when constructing a CompGraph out of the provided [`ONNX.GraphProto`](@ref).
"""
struct CompGraphBuilder
   g::ONNX.GraphProto
   sizes::Dict{String, <:Tuple}
   allnodes::Dict{String, OnnxNode}
   created::IdDict{OnnxNode, AbstractVertex}
   inputs::Vector{AbstractVertex}
end

function CompGraphBuilder(g::ONNX.GraphProto) 
   initdict = Dict(tp.name => tp for tp in g.initializer)
   CompGraphBuilder(g, sizes(g), output_to_node(g.node, initdict), IdDict{OnnxNode, AbstractVertex}(), AbstractVertex[])
end


function output_to_node(nodes, initdict)
   allnodes = Dict{String, OnnxNode}()
   for nodeproto in nodes
      ps = params(nodeproto, initdict)
      node = OnnxNode(nodeproto, ps)
      for outname in output(node)
         allnodes[outname] = node
      end
   end
   return allnodes
end

Broadcast.broadcastable(gb::CompGraphBuilder) = Ref(gb)
Broadcast.broadcastable(n::OnnxNode) = Ref(n)

node(name::String, gb::CompGraphBuilder, parent=nothing) = get!(gb.allnodes, name) do
   # TODO: Handle this in some other way
   # Create a fake node to make 1<->1 mapping with NaiveNASflux which uses a special vertex type as output
   inputnode = ONNX.NodeProto()
   inputnode.input = AbstractString[]
   inputnode.output = AbstractString[parent.name]
   inputnode.name = name
   inputnode.op_type= "Input"

   return OnnxNode(inputnode, ONNX.TensorProto[], Dict{Symbol, Any}(:size=>gb.sizes[name]))
end
nodes(gb::CompGraphBuilder) = values(gb.allnodes)
innames(n::OnnxNode) = innames(n.proto)
innames(n::ONNX.NodeProto) = innames(Val(optype(n)), n)
innames(::Val, n::ONNX.NodeProto) = input(n)[1:min(1, length(input(n)))]
innames(::Val{:Input}, ::ONNX.NodeProto) = []
innames(::Val{:Add}, n::ONNX.NodeProto) = input(n)
innames(::Val{:Mul}, n::ONNX.NodeProto) = input(n)
innames(::Val{:Concat}, n::ONNX.NodeProto) = input(n)

input(n::ONNX.NodeProto) = n.input
input(n::OnnxNode) = input(n.proto)

function outnames(n::ONNX.NodeProto, gb::CompGraphBuilder)
   allins = vcat(innames.(nodes(gb))...)
   return filter(oname -> oname in allins, output(n))
end
output(n::ONNX.NodeProto) = n.output
output(n::OnnxNode) = output(n.proto)

innodes(n::OnnxNode, gb::CompGraphBuilder) = innodes(n.proto, gb)
innodes(n::ONNX.NodeProto, gb::CompGraphBuilder) = map(ni -> node(ni, gb, n), innames(n))
outnodes(n::OnnxNode, gb::CompGraphBuilder) = outnodes(n.proto, gb)
function outnodes(n::ONNX.NodeProto, gb::CompGraphBuilder)
    onames = outnames(n, gb)
    filter(nn -> any(on -> on in nn.input, onames), gb.g.node)
end

optype(n::ONNX.NodeProto) = Symbol(n.op_type)
optype(n::OnnxNode) = optype(n.proto)

Flux.params(n::ONNX.NodeProto, initdict) = params(Val(optype(n)), n, initdict)
Flux.params(::Val, n::ONNX.NodeProto, initdict) = map(pname -> initdict[pname], setdiff(input(n), innames(n))) # Inputs which are not other vertices
Flux.params(n::OnnxNode) = n.params .|> array 