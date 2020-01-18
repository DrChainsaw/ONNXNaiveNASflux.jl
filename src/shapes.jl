

struct Shape1D <: FluxLayer end
NaiveNASflux.indim(::Shape1D) = 1
NaiveNASflux.outdim(::Shape1D) = 1
NaiveNASflux.actdim(::Shape1D) = 1
NaiveNASflux.actrank(::Shape1D) = 0


numpy2fluxdim(np_axis, v::AbstractVertex) = numpy2fluxdim(np_axis, 1 + NaiveNASflux.actrank(v)[1])
numpy2fluxdim(np_axis, ndims) = np_axis >= 0 ? ndims - np_axis : abs(np_axis)

flux2numpydim(dim, ndims) = ndims - dim

ndims_shape(shape::Tuple{<:Integer, <:FluxLayer}) = 1 + NaiveNASflux.actrank(shape[2])
ndims_shape(shape::NTuple{N, <:Integer}) where N = N
