# Nightmare inducing logic to combine/remove nodes, e.g layers and activation functions or reshape between recurrent layers and dense layers.

struct ActivationFunction end

struct ActivationLayer end

struct RecurrentLayer end

struct ElemwiseOp end

"""
   check_combine(gb::CompGraphBuilder, n::OnnxNode)

Return a potentially combined [`OnnxNode`](@ref) and its inputs.

Purpose is to handle e.g activation functions to layers which Flux has as member of the layer struct while ONNX has as a separate node.

Main motivation is an attempt to make serialization/deserialization "pure" in the sense that `g -> serialize -> deserialize === g`.

Another motivation is that mutation becomes a bit harder if things like global pooling and dropping of dimensions are separated into two vertices.
"""
check_combine(gb::CompGraphBuilder, n::OnnxNode) = check_combine(optrait(n), n, gb)

optrait(n::OnnxNode) = optrait(Val(optype(n)), n)
function optrait(vot::Val{ot}, n::OnnxNode) where ot
      ot in keys(actfuns) && return ActivationFunction()
      ot in keys(fluxrecurrentlayers) && return RecurrentLayer()
      ot in keys(fluxlayers) && return ActivationLayer()
      return vot
end
optrait(::Val{:Mul}, ::OnnxNode) = ElemwiseOp()
optrait(::Val{:Add}, ::OnnxNode) = ElemwiseOp()
optrait(::Val{:Div}, ::OnnxNode) = ElemwiseOp()

retnode(n, gb) = n, innodes(n, gb)
check_combine(x, n::OnnxNode, gb::CompGraphBuilder) = retnode(n, gb)

# Activation function: If there is exactly one input and that input is a layer with an activation function then combine them into one node
check_combine(::ActivationFunction, n::OnnxNode, gb::CompGraphBuilder) = check_actfun(n, gb, innodes(n, gb)...)


check_actfun(nact::OnnxNode, gb::CompGraphBuilder, args...) = retnode(nact, gb)
check_actfun(nact::OnnxNode, gb::CompGraphBuilder, nlayer::OnnxNode) = check_actfun(nact, gb, nlayer, optrait(nlayer))
check_actfun(nact::OnnxNode, gb::CompGraphBuilder, nlayer::OnnxNode, ::ActivationLayer) = wrapfrom(nact, nlayer, gb, :activation, actfuns)

function wrapfrom(nwrap::OnnxNode, nwrapped::OnnxNode, gb::CompGraphBuilder, attrib, from::AbstractDict=invariantops)
   @debug "Merge node $nwrap and node $nwrapped"
   nwrapped.attribute[attrib] = from[optype(nwrap)](nwrap.attribute, params(nwrap)...)
   return retnode(nwrapped, gb)
end

# Reshape
# 1. Check if input node is a global pool and if so combine them into one op
# 2. Check if input is from recurrent layers and if so remove the reshape as it is not needed
check_combine(::Val{:Reshape}, n::OnnxNode, gb::CompGraphBuilder) = check_reshape(n, gb, innodes(n, gb)...)

check_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, args...) = retnode(nreshape, gb)
check_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode) = check_reshape(nreshape, gb, innode, optrait(innode))

check_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode, ::Val{:GlobalAveragePool}) = wrapfrom(nreshape, innode, gb, :wrap, pseudotransparentops)
check_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode, ::Val{:GlobalMaxPool}) = wrapfrom(nreshape, innode, gb, :wrap, pseudotransparentops)

check_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, nrec::OnnxNode, ::RecurrentLayer) = check_recurrent_reshape(nreshape, gb, nrec)
check_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, nsqueeze::OnnxNode, ::Val{:Squeeze}) = check_recurrent_reshape(nreshape, gb, innodes(nsqueeze, gb)...)

check_recurrent_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, args...) = retnode(nreshape, gb)
check_recurrent_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode) = check_recurrent_reshape(nreshape, gb, innode, optrait(innode))

function check_recurrent_reshape(nreshape::OnnxNode, gb::CompGraphBuilder, nrec::OnnxNode, ::RecurrentLayer)
   outs = outnodes(nreshape, gb)
   # Maybe check that the reshape really is doing 3D -> 2D, but what else could it do in this case?
   if length(outs) > 0 && all(check_recurrent_reshape, Val.(optype.(outs)))
      @debug "Remove reshape found between $(outs) and $nrec"
      return check_combine(gb, innodes(nreshape, gb)[1])
   end
end

check_recurrent_reshape(::Val) = false
check_recurrent_reshape(::Val{:Gemm}) = true

# Squeeze
# 1. Check if input node is a global pool and if so combine them into one op
# 2.  Check if input is from recurrent layers and if so remove the squeeze as it is not needed
check_combine(::Val{:Squeeze}, n::OnnxNode, gb::CompGraphBuilder) = check_squeeze(n, gb, innodes(n, gb)...)


check_squeeze(nsqueeze::OnnxNode, gb::CompGraphBuilder, args...) = retnode(nsqueeze, gb)
check_squeeze(nsqueeze::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode) = check_squeeze(nsqueeze, gb, innode, optrait(innode))

check_squeeze(nsqueeze::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode, ::Val{:GlobalAveragePool}) = wrapfrom(nsqueeze, innode, gb, :wrap)
check_squeeze(nsqueeze::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode, ::Val{:GlobalMaxPool}) = wrapfrom(nsqueeze, innode, gb, :wrap)

function check_squeeze(nsqueeze::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode, ::RecurrentLayer)
   @debug "Remove squeeze after $innode"
   innode.attribute[SQUEEZED_RECURRENT_KEY] = true
   return retnode(innode, gb)
end

# For element wise operations, check if one of the inputs is a constant
check_combine(::ElemwiseOp, n::OnnxNode, gb::CompGraphBuilder) = check_elemwise(n, gb, innodes(n, gb))

function check_elemwise(nelemwise::OnnxNode, gb::CompGraphBuilder, innodes::AbstractVector{OnnxNode})
   ins_to_process = filter(!isnothing, map(innode -> check_elemwise(nelemwise, gb, innode, optrait(innode)), innodes))
   return nelemwise, ins_to_process
end

check_elemwise(::OnnxNode, ::CompGraphBuilder, innode::OnnxNode, ot) = innode

function check_elemwise(nelemwise::OnnxNode, gb::CompGraphBuilder, innode::OnnxNode, ::Val{:Constant})
   consts = get!(nelemwise.attribute, :Constant, [])
   push!(consts, sources[optype(innode)](innode.attribute, params(innode)...))
   return nothing
end
