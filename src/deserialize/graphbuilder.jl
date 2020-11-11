
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

sizes(mp::ONNX.ModelProto) = sizes(mp.graph)
sizes(gp::ONNX.GraphProto) = Dict((name.(gp.input) .=> size.(gp.input))..., (name.(gp.output) .=> size.(gp.output))...)


function output_to_node(nodes, initdict)
   allnodes = Dict{String, OnnxNode}()
   for nodeproto in nodes
      # Past this point we get hard to interpret errors if optype is not supported
      @assert optype(nodeproto) in keys(verts) "Optype $(optype(nodeproto)) not supported!"
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

node(nodename::String, gb::CompGraphBuilder, parent=nothing) = get!(gb.allnodes, nodename) do
   # TODO: Handle this in some other way
   # Create a fake node to make 1<->1 mapping with NaiveNASflux which uses a special vertex type as output
   inputnode = ONNX.NodeProto()
   inputnode.input = AbstractString[]
   inputnode.output = AbstractString[name(parent)]
   inputnode.name = nodename
   inputnode.op_type= "Input"

   pnode = get(gb.allnodes, name(parent), nothing)
   ltypes = find_valid_fluxlayertype(pnode, gb) |> Tuple

   inshape = gb.sizes[nodename]
   ltype = select_layertype(nodename, inshape, ltypes)

   return OnnxNode(inputnode, ONNX.TensorProto[], Dict{Symbol, Any}(:size=>inshape, :ltype=>ltype))
end

find_valid_fluxlayertype(::Nothing, gb, direction=outnames) = []
function find_valid_fluxlayertype(n::OnnxNode, gb, direction=outnames)
   lt = fluxlayertypes[optype(n)](n.params...)
   lt isa FluxParLayer && return [lt]
   return mapreduce(vcat, direction(n, gb); init=[]) do outname
      outnode = get(gb.allnodes, outname, nothing)
      find_valid_fluxlayertype(outnode, gb, direction)
   end
end

function select_layertype(inname, inshape, lts::Tuple)
   ltsvalid = filter(lts) do lt
      length(inshape) == length(shape(lt, 0))
   end

   length(ltsvalid) == 1 && return ltsvalid[1]
   length(ltsvalid) == 0 && return guess_layertype(length(inshape))
   @warn "Multiple layertypes found for input $inname with shape $inshape: $(ltsvalid)! Graph mutation near this vertex might fail!"
   return first(ltsvalid)
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

outnames(n::OnnxNode, gb) = outnodes(n.proto, gb)
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