
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

CompGraphBuilder(g::ONNX.GraphProto) = CompGraphBuilder(g, Dict()) # To avoid ambigity
function CompGraphBuilder(g::ONNX.GraphProto, insizes::Tuple...) 
   @assert length(insizes) <= length(g.input) "Too many inputs supplied! Got $(length(insizes)) while model inputs are $(name.(g.input))"
   CompGraphBuilder(g, map(=>, name.(g.input), insizes)...)
end
function CompGraphBuilder(g::ONNX.GraphProto, insizes::Pair{String, <:Tuple}...)
   @assert length(insizes) <= length(g.input) "Too many inputs supplied! Got $(length(insizes)) while model inputs are $(name.(g.input))"
   @assert all(n -> n in name.(g.input), first.(insizes)) "Input names does not match! Got $(first.(insizes)) to model with input names $(name.(g.input))"
   CompGraphBuilder(g, merge(sizes(g), Dict(insizes...)))
end
   
function CompGraphBuilder(g::ONNX.GraphProto, inoutsizes::AbstractDict)
   allsizes = merge(sizes(g), inoutsizes) |> clean_size
   initdict = Dict(tp.name => tp for tp in g.initializer)
   CompGraphBuilder(g, allsizes, output_to_node(g.node, initdict), IdDict{OnnxNode, AbstractVertex}(), AbstractVertex[])
end

sizes(mp::ONNX.ModelProto) = sizes(mp.graph)
sizes(gp::ONNX.GraphProto) = Dict((name.(gp.input) .=> size.(gp.input))..., (name.(gp.output) .=> size.(gp.output))...)

clean_size(d::AbstractDict) = Dict(k => clean_size(v) for (k,v) in d)
clean_size(t::Tuple) = int_size.(t)
clean_size(::Missing) = tuple()
int_size(i::Integer) = i
int_size(::Any) = 0

function output_to_node(nodes, initdict)
   allnodes = Dict{String, OnnxNode}()
   for nodeproto in nodes
      # Past this point we get hard to interpret errors if optype is not supported
      if optype(nodeproto) ∉ keys(verts) 
         throw(OpNotSupportedError(optype(nodeproto)))
      end
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
   inputnode = ONNX.NodeProto(;
                           input = AbstractString[],
                           output = AbstractString[name(n) for n in values(gb.allnodes) if nodename in innames(n)],
                           name = nodename,
                           op_type= "Input")

   # We want nodes as well as layertypes in case we need to infer the size from the selected node
   # Unfortunate failure case: If the only way to see a valid fluxlayertype is through concatenation we will fail to infer the shape.
   # This is because find_valid_fluxlayertype must also return a node for which the input size can be used as the size of inputnode, 
   # and it is not capable of doing so if we pass through concatenation. Might need to split up layertypes and sizes and try to 
   # connect them later which just feels like a huge hassle right now. Another option is perhaps to track if we pass throgh concatenation
   # and return some other object which deals with the concatenation, either by failing, or in cases when it is possible to calculate 
   # backwards (e.g. inputnode concatenated with a node of known size).
   names_to_search = filter(n -> n in keys(gb.allnodes), inputnode.output)
   node_to_ltypes = mapreduce(outname -> find_valid_fluxlayertype(gb.allnodes[outname], gb), vcat, names_to_search; init=[]) |> Tuple

   # gb.sizes can be missing, 0 or some tuple where the nout dimension is missing
   # If this is the case we try to infer it from one of node_to_ltypes
   inshape, ltype= select_layertype(nodename, gb.sizes[nodename], node_to_ltypes)

   return OnnxNode(inputnode, ONNX.TensorProto[], Dict{Symbol, Any}(:size=>inshape, :ltype=>ltype))
end

find_valid_fluxlayertype(::Nothing, gb, seen=[]) = []
function find_valid_fluxlayertype(n::OnnxNode, gb, seen=[])
   n in seen && return []
   push!(seen, n)

   lt = fluxlayertypes[optype(n)](n.params...)
   lt isa FluxParLayer && return [n => generic(lt)]
   stop_search(optype(n)) && return []

   return mapreduce(vcat, vcat(outnames(n, gb), innames(n)); init=[]) do outname
      outnode = get(gb.allnodes, outname, nothing)
      find_valid_fluxlayertype(outnode, gb, seen)
   end
end

generic(::FluxConvolutional{N}) where N = GenericFluxConvolutional{N}()
generic(::Flux2D) = GenericFlux2D()
generic(::FluxRecurrent) = GenericFluxRecurrent()
generic(lt) = lt

stop_search(ot) = false
stop_search(ot::Symbol) = stop_search(Val{ot})
stop_search(::Type{Val{:Reshape}}) = true
stop_search(::Type{Val{:Flatten}}) = true
stop_search(::Type{Val{:Squeeze}}) = true
stop_search(::Type{Val{:ReshapeMean}}) = true
# This is a bit unfortunate as we can still infer the layertype, just not the size. They are however connected right now
# and decoupling them would be a bit of a hassle...
stop_search(::Type{Val{:Concat}}) = true

function select_layertype(inname, inshape, node_to_lts::Tuple)
   node_to_lts_valid = filter(node_to_lts) do n2lt
      length(inshape) == 0 || length(inshape) == length(shape(last(n2lt), 0))
   end
   
   insize_to_lts_valid = unique(fix_invalid_insize.(Ref(inshape), node_to_lts_valid))

   length(insize_to_lts_valid) == 1 && return insize_to_lts_valid[1]
   length(insize_to_lts_valid) == 0 && return inshape, guess_layertype(length(inshape))
   @warn "Multiple layertypes found for input $inname with shape $inshape: $(last.(insize_to_lts_valid))! Graph mutation near this vertex might fail!"
   return first(insize_to_lts_valid)
end

fix_invalid_insize(::Missing, (node,lt)::Pair) = shape(lt, fluxlayer_nin(node)) => lt
fix_invalid_insize(::Tuple{}, node_to_lt::Pair) = fix_invalid_insize(missing, node_to_lt)
fix_invalid_insize(inshape::Tuple, (node,lt)::Pair) = fix_invalid_insize(inshape[actdim(lt)], node=>lt)
fix_invalid_insize(insize::Integer, node_to_lt) = insize > 0 ? shape(last(node_to_lt), insize) => last(node_to_lt) : fix_invalid_insize(missing, node_to_lt) 

# Not beautiful that we instantiate a whole layer and then throw it away just to measure nout, but we generally don't do this very often
fluxlayer_nin(node::OnnxNode) = fluxlayers[optype(node)](node.attribute, params(node)...) |> nin |> first

nodes(gb::CompGraphBuilder) = values(gb.allnodes)

# This design is not good! 
# innames are supposed to give the names of other nodes in the graph which are input (as opposed to input which in Flux layers is considered parameters).
# Here we kind of hardcode per OpType which inputs we expect are parameters (or else you typically can't make a Flux layer out of it).
# Can't we just filter out names which are part of initializers? Perhaps one also need to
# propagate constants eventually, but this is anyways not supported in other parts.
# If it turns out that this can't be made into a Flux layer we can just throw a more sensible
# exception later
innames(n::OnnxNode) = innames(n.proto)
innames(n::ONNX.NodeProto) = innames(Val(optype(n)), n)
innames(::Val, n::ONNX.NodeProto) = input(n)[1:min(1, length(input(n)))]
innames(::Val{:Input}, ::ONNX.NodeProto) = []
innames(::Val{:Add}, n::ONNX.NodeProto) = input(n)
innames(::Val{:Mul}, n::ONNX.NodeProto) = input(n)
innames(::Val{:Div}, n::ONNX.NodeProto) = input(n)
innames(::Val{:Concat}, n::ONNX.NodeProto) = input(n)
innames(::Val{:MatMul}, n::ONNX.NodeProto) = input(n)[1:1]

input(n::ONNX.NodeProto) = n.input
input(n::OnnxNode) = input(n.proto)

outnames(n::OnnxNode, gb) = outnames(n.proto, gb)
outnames(n::ONNX.NodeProto, gb::CompGraphBuilder) = name.(outnodes(n, gb))

output(n::ONNX.NodeProto) = n.output
output(n::OnnxNode) = output(n.proto)

innodes(n::OnnxNode, gb::CompGraphBuilder) = innodes(n.proto, gb)
innodes(n::ONNX.NodeProto, gb::CompGraphBuilder) = map(ni -> node(ni, gb, n), innames(n))
outnodes(n::OnnxNode, gb::CompGraphBuilder) = outnodes(n.proto, gb)
function outnodes(n::ONNX.NodeProto, gb::CompGraphBuilder)
   allins = mapreduce(innames, vcat, nodes(gb))
   onames = filter(oname -> oname in allins, output(n))
   filter(nn -> any(on -> on in nn.input, onames), gb.g.node)
end

optype(n::ONNX.NodeProto) = Symbol(n.op_type)
optype(n::OnnxNode) = optype(n.proto)

Flux.params(n::ONNX.NodeProto, initdict) = params(Val(optype(n)), n, initdict)
Flux.params(::Val, n::ONNX.NodeProto, initdict) = map(pname -> initdict[pname], setdiff(input(n), innames(n))) # Inputs which are not other vertices
Flux.params(n::OnnxNode) = n.params .|> array 