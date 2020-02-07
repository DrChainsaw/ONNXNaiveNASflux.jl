

struct Shape1D <: FluxLayer end
NaiveNASflux.indim(::Shape1D) = 1
NaiveNASflux.outdim(::Shape1D) = 1
NaiveNASflux.actdim(::Shape1D) = 1
NaiveNASflux.actrank(::Shape1D) = 0


numpy2fluxdim(np_axis, v::AbstractVertex) = numpy2fluxdim(np_axis, 1 + NaiveNASflux.actrank(v)[1])
numpy2fluxdim(np_axis, ndims) = np_axis >= 0 ? ndims - np_axis : abs(np_axis)

flux2numpydim(dim, ndims) = ndims - dim

function shape(v::AbstractVertex)
    outshape = shape(layertype(v), nout(v))
    ismissing(outshape) && return first(unique(shape.(inputs(v))))
    return outshape
end
shape(::FluxLayer, outsize) = missing
shape(::Shape1D, outsize) = (outsize,)
shape(::FluxDense, outsize) = (outsize, missing)
shape(::FluxConvolutional{N}, outsize) where N = ((missing for _ in 1:N)..., outsize, missing)
shape(::FluxRecurrent, outsize) = (outsize, missing, missing)

aggshape(f, d::Number...) = f(d...)
aggshape(f, d...) = missing

rmdims(t::Tuple, dim::Integer) = t[1:end .!= dim]
rmdims(t::Tuple, dims) = Tuple(t[i] for i in 1:length(t) if i ∉ dims)

function guess_layertype(ndims::Integer)
    ndims <= 1 && return Shape1D()
    ndims == 2 && return FluxDense()
    ndims == 3 && return FluxRnn()
    return FluxConv{ndims-2}()
end

flipweights(l, w) = w
flipweights(::FluxConvolutional{N}, w) where N = w[(size(w,i):-1:1 for i in 1:N)..., :, :]
flipweights(::FluxRnn, w, hsize) = w
function flipweights(::FluxLstm, w, hsize)
    input = Flux.gate(w, hsize, 1)
    forget = Flux.gate(w, hsize, 2)
    cell = Flux.gate(w, hsize, 3)
    output = Flux.gate(w, hsize, 4)
    return vcat(input, output, forget, cell)
end

unflipweights(::FluxRnn, w, hsize) = w
function unflipweights(::FluxLstm, w, hsize)
    input = Flux.gate(w, hsize, 1)
    output = Flux.gate(w, hsize, 2)
    forget = Flux.gate(w, hsize, 3)
    cell = Flux.gate(w, hsize, 4)
    return vcat(input, forget, cell, output)
end

outshape(l, s) = outshape(layertype(l), l, s)

outshape(lt::FluxDense, l, s) = (nout(l), s[end])
function outshape(lt::FluxConvolutional{N}, l, s) where N
    p = length(l.pad) == N ? 2 .* l.pad : l.pad[1:2:end] .+ l.pad[2:2:end]
    k = size(weights(l))[1:N]
    d = l.dilation
    stride = l.stride

    o = map(zip(1:N, s)) do (i, si)
        # Conv arithmetic from https://arxiv.org/pdf/1603.07285.pdf
        aggshape(x -> (x + p[i] - k[i] - (k[i] - 1)*(d[i] - 1)) ÷ stride[i] + 1, si)
    end

    return (o..., nout(l), s[end])
end

function outsize(l::Union{Flux.MaxPool{N}, Flux.MeanPool{N}}, s) where N
    p = length(l.pad) == N ? 2 .* l.pad : l.pad[1:2:end] .+ l.pad[2:2:end]
    k = l.k
    stride = l.stride

    o = map(zip(1:N, s)) do (i, si)
        # Conv arithmetic from https://arxiv.org/pdf/1603.07285.pdf
        aggshape(x -> (x + p[i] - k[i]) ÷ stride[i] + 1, si)
    end

    return (o..., nout(l), s[end])
end
