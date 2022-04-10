const sources = Dict{Symbol, Any}()
const actfuns = Dict{Symbol, Any}()
const rnnactfuns = Dict{Symbol, Any}() # Recurrent layers have activation functions as attributes and use different parameter names compared to their respective operations.
const actlayers = Dict{Symbol, Any}()
const fluxlayers = Dict{Symbol, Any}()
const fluxrecurrentlayers = Dict{Symbol, Any}()
const invariantops = Dict{Symbol, Any}()
const pseudotransparentops = Dict{Symbol, Any}()
const verts = Dict{Symbol, Any}()
const fluxlayertypes = Dict{Symbol, Any}()

# Rundown of the basic idea here:

# Aspect 1
# ONNX does not have activation functions as an attribute to its layers but rather represents them as a separate node
# This would indeed be workable, but...
# 1. It is a bit annoying that model -> serialize -> deserialize does not result in the exact same thing
# 2. If one wants to use the mutation functionality of NaiveNASflux it might not be desirable to have activation
#    functions as separate vertices in the graph as this invites for things like inserting something else between
#    the layer and its activation function.

# To be able to have activation functions back inside their layers when deserializing, whenever an op which is a key
# in actlayers is encountered there is a "lookahead" to see if the op of the next node is in actfuns. If it is, the
# two ops will be merged into one vertex containing the layer and its activation function.
# A very similar thing is done for global pooling operations followed by squeeze or reshape.

# Aspect 2
# The vertices of NaiveNASflux require a few inputs when creating them. One in particular is knowledge of the size
# trait which is obviously not possible to obtain from the ONNX data. In order to spare users from having to supply
#  this extra input with each operation there is one dict per "general type".

# As NaiveNASflux already has the knowledge what is needed for all layers in Flux, they have their own dict
#  (fluxlayers) which just outsources the vertex creation to NaiveNASflux. Note that all actlayers are inserted
# in this dict.

# Functions which always produce the same number of outputs as inputs and are not defined in Flux, e.g.
#  GlobalAveragePool end up in invariantops.

# Functions which have dedicated vertex construction methods, such as Concat and Add end up in verts.

struct OpNotSupportedError <: Exception
    msg::String
end
OpNotSupportedError(op_type::Symbol) = OpNotSupportedError(string("Operation type ", op_type, " not supported!"))
Base.showerror(io::IO, e::OpNotSupportedError) = print(io, "OpNotSupportedError: ", e.msg)

sources[:Constant] = params -> constant(Val.(keys(params))..., values(params)...)
constant(::Val{:value}, val::ONNX.TensorProto) = val |> array
constant(::Val{:value}, val) = val

actfuns[:Relu] = params -> Flux.relu

actfuns[:Elu] = function(params)
    α = get(params, :alpha, 1)
    return x -> Flux.elu(x, oftype(x, α))
end
rnnactfuns[:Elu] = (ind, params) -> actfuns[:Elu](Dict(:alpha => get(params, :activation_alpha, ntuple(i -> 1, ind))[ind]))

actfuns[:Selu] = function(params)
    haskey(params, :alpha) || haskey(params, :gamma) && return Flux.selu
    γ = get(params, :gamma, Float32(1.05070102214813232421875))
    α = get(params, :alpha, Float32(1.67326319217681884765625))
    return x -> selu(x, oftype(x, γ), oftype(x, α))
end
Flux.selu(x, γ, α) = γ * ifelse(x > 0, x/1, α * (exp(x) - 1))

actfuns[:Tanh] = params -> tanh
rnnactfuns[:Tanh] = (ind, params) -> tanh


mrev(x) = x
mrev(x::AbstractVector) = reverse(x)
prev(x) = x
prev(x::AbstractVector) = reshape(permutedims(reverse(reshape(x, length(x) ÷ 2,:);dims=1)),:)


# mrev = maybe reverse. prev = rearrange padding, e.g. (1,2,1,2) => (2,2,1,1) or (1,2,3,1,2,3) => (3,3,2,2,1,1)
_akpsd(params) = get(params, :activation, identity), mrev(get(params, :kernel_shape, 1)), prev(get(params, :pads, 0)), mrev(get(params, :strides, 1)), mrev(get(params, :dilations, 1))
akpsd(params) = a2t.(_akpsd(params))
a2t(x) = x
a2t(a::AbstractArray) = Tuple(a)

actlayers[:Conv] = function(params, weight::AbstractArray{T, N}, bias=false) where {T, N}
    a,_,p,s,d = akpsd(params)
    @assert get(params, :group, 1) == 1 "Group size not supported!" # TODO
    return Conv(flipweights(FluxConv{N-2}(), weight), bias, a, pad=p, stride=s, dilation=d)
end
fluxlayertypes[:Conv] = (weight, bias=nothing) -> FluxConv{length(size(weight))-2}()

biasarray(b::Number, esize, β) = b
biasarray(b::AbstractArray, esize, β) = length(b) === 1 ? repeat(β .* vec(b), esize) : β .* reshape(b, :)
biasarray(b::Number, esize, β) = repeat([β * b], esize)

actlayers[:Gemm] = function(params, weight::AbstractArray{T, N}, bias=false) where {T,N}
    act = get(params, :activation, identity)
    wt = Bool(get(params, :transB, 0)) ? permutedims : identity
    α = get(params, :alpha, 1)
    β = get(params, :beta, 1)

    weight = α .* wt(weight)
    bias = biasarray(bias, size(weight, 1), β)

    return Dense(weight, bias, act)
end
fluxlayertypes[:Gemm] = (pars...) -> FluxDense()

actlayers[:BatchNormalization] = function(params, γ, β, μ, σ²)
    λ = get(params, :activation, identity)
    ϵ = get(params, :epsilon, 1f-5)
    momentum = get(params, :momentum, 9f-1)

    return BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, true, true, nothing, length(γ))
end
fluxlayertypes[:BatchNormalization] = (pars...) -> FluxBatchNorm()


default_Wb_Rb(Wh_WBh) = fill!(similar(Wh_WBh, (size(Wh_WBh, 2) * 2, size(Wh_WBh, 3))), 0)
default_init_h(Wb_Rb, sc) = fill!(similar(Wb_Rb, (size(Wb_Rb,1) ÷ sc, size(Wb_Rb,2))), 0)
# TODO when https://github.com/FluxML/Flux.jl/issues/1279 is resolved default_init_h(Wh_WBh, sc) = fill!(similar(Wh_WBh, (size(Wh_WBh, 2) ÷ sc, size(Wh_WBh, 3))), 0)

fluxrecurrentlayers[:RNN] = function(params, Wi_WBi, Wh_WBh, Wb_Rb=default_Wb_Rb(Wh_WBh), seqlen=[], h3d = default_init_h(Wb_Rb, 2))
    @assert size(Wi_WBi, 3) == 1 "Num directions must be 1! Bidirectional (num directions = 2) not supported!" # TODO: Add...

    Wi,Wh,b,h = recurrent_arrays(FluxRnn(), Wi_WBi, Wh_WBh, Wb_Rb, h3d)
    act = rnnactfuns[Symbol(get(params, :activations, ["Tanh"])[])](1, params)
    cell = Flux.RNNCell(act, Wi, Wh, b, fill!(similar(h), 0))
    return Flux.Recur(cell, h)
end
fluxlayertypes[:RNN] = (pars...) -> FluxRnn()


fluxrecurrentlayers[:LSTM] = function(params, Wi_WBi, Wh_WBh, Wb_Rb=default_Wb_Rb(Wh_WBh), seqlen=[1], h3d = default_init_h(Wb_Rb, 8), c3d=default_init_h(Wb_Rb,8), peep=nothing)
    @assert size(Wi_WBi, 3) == 1 "Num directions must be 1! Bidirectional (num directions = 2) not supported!" # TODO: Add...
    @assert isnothing(peep) "Peepholes not supported!" # Or?
    Wi,Wh,b,h,c = recurrent_arrays(FluxLstm(), Wi_WBi, Wh_WBh, Wb_Rb, h3d, c3d)
    # Flux only supports default activation functions
    # We can only check that given values doesn't deviate
    supported = [:Sigmoid, :Tanh, :Tanh]
    acts = get(params, :activations, supported)
    @assert all(zip(supported, acts)) do (e,a)
        e == a
    end "Got unsupported activation function: $acts"

    # b, h and c must all be of the same type when creating a cell, but
    # it is actually Recur which has the state
    cell = Flux.LSTMCell(Wi, Wh, b, (fill!(similar(h), 0), fill!(similar(c), 0)))
    return Flux.Recur(cell, (h, c))
end
fluxlayertypes[:LSTM] = (pars...) -> FluxLstm()

function recurrent_arrays(lt, Wi_WBi, Wh_WBh, Wb_Rb, h3ds...)
    # ONNX weights are on the form [num_directions, hidden_size, input_size] (where num_directions is 2 for bidirectional else 1)
    # Flux weights are of shape [hidden_size, input_size]
    # To spice things up a bit, all julia arrays are loaded in reverse order, i.e we get an array with the arrangement [input_size, hidden_size, num_directions].
    # First remove the num_directions dimension, then transpose into the correct shape
    hsize = size(Wh_WBh, 1)
    Wi = unflipweights(lt, permutedims(dropdims(Wi_WBi, dims=3)), hsize)
    Wh = unflipweights(lt, permutedims(dropdims(Wh_WBh, dims=3)), hsize)
    b = Wb_Rb isa Number ? Wb_Rb : dropdims(unflipweights(lt, sum(reshape(Wb_Rb, :, 2), dims=2), hsize),dims=2)
    return Wi, Wh, b, h3ds...
end

fluxlayers[:MaxPool] = function(params)
    _,k,p,s,_ = akpsd(params)
    return MaxPool(k, pad=p, stride=s)
end
fluxlayertypes[:MaxPool] = (pars...) -> FluxPoolLayer()


fluxlayers[:AveragePool] = function(params)
    _,k,p,s,_ = akpsd(params)
    return MeanPool(k, pad=p, stride=s)
end
fluxlayertypes[:AveragePool] = (pars...) -> FluxPoolLayer()


fluxlayers[:Dropout] = params -> Dropout(get(params, :ratio, 0.5))
fluxlayertypes[:Dropout] = (pars...) -> FluxDropOut()


invariantops[:GlobalAveragePool] = function(params)
    wrap = get(params, :wrap, identity)
    return x -> globalmeanpool(x, wrap)
end
fluxlayertypes[:GlobalAveragePool] = (pars...) -> FluxPoolLayer()

function globalmeanpool(x::AbstractArray{T,N}, wrap) where T where N
    wrap(MeanPool(size(x)[1:N-2])(x))
end

invariantops[:GlobalMaxPool] = function(params)
    wrap = get(params, :wrap, identity)
    return x -> globalmaxpool(x, wrap)
end
fluxlayertypes[:GlobalMaxPool] = (pars...) -> FluxPoolLayer()

function globalmaxpool(x::AbstractArray{T,N}, wrap) where T where N
    wrap(MaxPool(size(x)[1:N-2])(x))
end

invariantops[:Squeeze] = function(params)
    np_axes = get(params, :axes, missing)
    dimfun = ismissing(np_axes) ? x -> Tuple(findall(i -> i == 1, size(x))) : x -> Tuple(numpy2fluxdim.(np_axes, ndims(x)))
    return x -> dropdims(x, dims=dimfun(x))
end

invariantops[:ReduceMean] = function(params)
    np_axes = get(params, :axes, missing)
    keepdims = Bool(get(params, :keepdims, 1))

    dimexp =
    if keepdims && ismissing(np_axes)
        # As mean returns a scalar when no dimensions are provided
        expanddims
    elseif !keepdims
        (out, x, dims) -> dropdims(out, dims=dims)
    else
        (out, x, dims) -> out
    end

    ismissing(np_axes) && return x -> dimexp(mean(x), x, missing)

    return function(x)
        dims = Tuple(numpy2fluxdim.(np_axes, ndims(x)))
        out = mean(x, dims=dims)
        return dimexp(out, x, dims)
    end
end
expanddims(out, x, dims) = fill(out, ntuple(i -> 1, ndims(x)))

invariantops[:Softmax] = params -> x -> onnxsoftmax(x; np_axis = get(params, :axis, 1))

function onnxsoftmax(x::AbstractArray{T, 2}; np_axis=1) where T
    dim = numpy2fluxdim(np_axis, 2)
    Flux.softmax(x; dims=dim)
end
function onnxsoftmax(x::AbstractArray{T, N}; np_axis=1) where {T,N}
    dim = numpy2fluxdim(np_axis, N)
    sz = size(x)
    reshape(Flux.softmax(reshape(x, prod(sz[1:dim]), :)), sz...)
end

pseudotransparentops[:Reshape] = function(params, shape)
    shape_t = Tuple(reverse(replace(shape, -1 => Colon())))
    return MeasureNout(Reshape(shape_t))
end

pseudotransparentops[:Flatten] = function(params)
    dim = -get(params,:axis, 1)
    return MeasureNout(Flatten(dim))
end


verts[:Input] = function(name, inputs, params; kwargs...)
    inshape = params[:size]
    ltype = params[:ltype]
    indims = length(inshape)

    insize = indims > 0 ? inshape[max(1, actdim(ltype))] : 1 # assume scalar
    return inputvertex(name, insize, ltype)
end

verts[:Add] = (name, inputs, params; kwargs...) -> elemwisevertex(name, inputs, params, +, 0; kwargs...)
verts[:Mul] = (name, inputs, params; kwargs...) -> elemwisevertex(name, inputs, params, *, 1; kwargs...)
verts[:Div] = (name, inputs, params; kwargs...) -> elemwisevertex(name, inputs, params, /, 1; kwargs...)

function elemwisevertex(name, inputs, params, op, id; traitdecoration=identity, layerfun=identity, kwargs...)
    c = reduce((c1,c2) -> op.(c1, c2), get(params, :Constant, id))
    c = length(c) == 1 ? c[] : c
    let cc = c
        opp, wrap = cc == id ? (op, layerfun) : (identity, f -> layerfun((x...) -> op.(cc, x...)))
        conf = VertexConf(traitdecoration = named(name) ∘ traitdecoration, outwrap = wrap, kwargs...)
        return NaiveNASlib.elemwise(opp, conf, inputs...)
    end
end


verts[:Concat] = function(name, inputs, params; traitdecoration=identity, layerfun=identity, kwargs...)
    dims = numpy2fluxdim(params[:axis], inputs[1])
    return conc(inputs..., dims=dims, traitdecoration = named(name) ∘ traitdecoration, outwrap=layerfun, kwargs...)
end

# Without parameters it needs its own type as well as constraints for propagation of size changes
matmul_op(name, inputs::AbstractVector{<:AbstractVertex}, params::AbstractDict; kwargs...) = throw(OpNotSupportedError("MatMul without parameter not supported!"))
matmul_op(name, inputs::AbstractVector{<:AbstractVertex}, params::AbstractDict, weight; kwargs...) = fluxvertex(name, Dense(weight, false, identity), inputs...; kwargs...)

verts[:MatMul] = matmul_op

function refresh()
    for (s, f) in actlayers
        fluxlayers[s] = f
    end

    for (s, f) in fluxrecurrentlayers
        fluxlayers[s] = f
    end

    for (s, f) in actfuns
        invariantops[s] = function(args...;kwargs...)
                                actfun = f(args...; kwargs...)
                                return x -> actfun.(x)
                            end
    end

    for (s, f) in fluxlayers
        verts[s] = (name, inputs, args...;kwargs...) -> fluxvertex(name, f(args...), inputs...; kwargs...)
    end

    for (s, f) in invariantops
        verts[s] = (name, inputs, args...;traitdecoration=identity, layerfun=identity, kwargs...) -> invariantvertex(layerfun(f(args...)), inputs...; traitdecoration = named(name) ∘ traitdecoration, kwargs...)
    end

    for (s,f) in pseudotransparentops
        verts[s] = function(name, inputs, args...;traitdecoration=identity, layerfun=identity, kwargs...)
            comp = f(args...)
            return absorbvertex(layerfun(comp), inputs...; traitdecoration = named(name) ∘ traitdecoration ∘ SizePseudoTransparent, kwargs...)
        end
    end

    for (s,f) in sources
        verts[s] = function(name, inputs, args...;kwargs...)
            @assert isempty(inputs) "Source of type $s got inputs $(inputs)!"
            return sourcevertex_with_outputs(f(args...), name)
        end
    end

    for s in keys(verts)
        get!(fluxlayertypes, s, (args...) -> missing)
    end

end

refresh()

list_supported_ops(io::IO=stdout) = foreach(ot -> println(io, ot), filter(ot -> ot != :Input, sort(collect(keys(verts)))))
