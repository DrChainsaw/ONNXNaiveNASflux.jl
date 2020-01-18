const actfuns = Dict{Symbol, Any}()
const actlayers = Dict{Symbol, Any}()
const fluxlayers = Dict{Symbol, Any}()
const invariantops = Dict{Symbol, Any}()
const verts = Dict{Symbol, Any}()

actfuns[:Relu] = params -> Flux.relu

_akpsd(params) = get(params, :activation, identity), get(params, :kernel_shape, 1), get(params, :pads, 0), get(params, :strides, 1), get(params, :dilations, 1)
akpsd(params) = a2t.(_akpsd(params))
a2t(x) = x
a2t(a::AbstractArray) = Tuple(a)

actlayers[:Conv] = function(params, weight::AbstractArray{T, N}, bias=zeros(T, size(weight, outdim(FluxConv{N-2}())))) where {T, N}
    a,_,p,s,d = akpsd(params)
    return Conv(weight, bias, a, pad=p, stride=s, dilation=d)
end

actlayers[:Gemm] = function(params, weight::AbstractArray{T, N}, bias=zeros(T, size(weight, outdim(FluxDense())))) where {T,N}
    act = get(params, :activation, identity)
    wt = Bool(get(params, :transB, 0)) ? permutedims : identity
    α = get(params, :alpha, 1)
    β = get(params, :beta, 1)
    return Dense(α * wt(weight), β * bias, act)
end

actlayers[:BatchNormalization] = function(params, γ, β, μ, σ²)
    λ = get(params, :activation, identity)
    ϵ = get(params, :epsilon, 1f-5)
    momentum = get(params, :momentum, 9f-1)

    return BatchNorm(λ, β, γ, μ, σ², ϵ, momentum)
end

fluxlayers[:MaxPool] = function(params)
    _,k,p,s,_ = akpsd(params)
    return MaxPool(k, pad=p, stride=s)
end

fluxlayers[:AveragePool] = function(params)
    _,k,p,s,_ = akpsd(params)
    return MeanPool(k, pad=p, stride=s)
end

fluxlayers[:Dropout] = params -> Dropout(get(params, :ratio, 0.5))

invariantops[:GlobalAveragePool] = function(params)
    wrap = get(params, :wrap, identity)
    return x -> globmeanpool(x, wrap)
end
function globmeanpool(x::AbstractArray{T,N}, wrap) where T where N
    wrap(MeanPool(size(x)[1:N-2])(x))
end

invariantops[:Reshape] = function(params, shape)
    shape_t = Tuple(reverse(shape))
    any(s -> s == 0 || s == -1, shape_t) && return x -> reshape_keepshape(x, shape_t)
    return x -> reshape(x, shape_t)
end
function reshape_keepshape(x, shape)
    offs = ndims(x) - length(shape)
    newshape = map(enumerate(shape)) do (ind, new)
        new == -1 && return Colon()
        new == 0 && return size(x, ind+offs)
        return new
    end
    return reshape(x, newshape...)
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
        (out, x, dims) -> fill(out, ntuple(i -> 1, ndims(x)))
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

verts[:Input] = function(name, inputs, params; kwargs...)
    inshape = params[:size]
    indims = length(inshape)

    if indims == 1
        return inputvertex(name, inshape[1], Shape1D())
    end

    if indims == 2
        return inputvertex(name, inshape[1], FluxDense())
    end
    return inputvertex(name, inshape[actdim(indims)], FluxConv{indims-2}())
end

verts[:Add] = function(name, inputs, params; conf=VertexConf())
    td = conf.traitdecoration
    nconf = @set conf.traitdecoration = t -> NamedTrait(td(t), name)
    return NaiveNASlib.elemwise(+, nconf, inputs...)
end

verts[:Concat] =  function(name, inputs, params; traitdecoration=identity, kwargs...)
    dims = numpy2fluxdim(params[:axis], inputs[1])
    return conc(inputs..., dims=dims, traitdecoration = t -> NamedTrait(traitdecoration(t), name), kwargs...)
end


function refresh()
    for (s, f) in actlayers
        fluxlayers[s] = f
    end

    for (s, f) in actfuns
        invariantops[s] = f
    end

    for (s, f) in fluxlayers
        verts[s] = (name, inputs, args...;kwargs...) -> mutable(name, f(args...), inputs...; kwargs...)
    end

    for (s, f) in invariantops
        verts[s] = (name, inputs, args...;traitdecoration=identity, kwargs...) -> invariantvertex(f(args...), inputs...; traitdecoration = t -> NamedTrait(traitdecoration(t), name), kwargs...)
    end
end

refresh()
