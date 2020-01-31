

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
shape(::FluxRecurrent, outsize) = (missing, outsize, missing)

rmdims(t::Tuple, dim::Integer) = t[1:end .!= dim]
rmdims(t::Tuple, dims) = Tuple(t[i] for i in 1:length(t) if i ∉ dims)

function guess_layertype(ndims)
    ndims <= 1 && return Shape1D()
    ndims == 2 && return FluxDense()
    return FluxConv{ndims-2}()
end

flipweights(l, w) = w
flipweights(::FluxConvolutional{N}, w) where N = w[(size(w,i):-1:1 for i in 1:N)..., :, :]
