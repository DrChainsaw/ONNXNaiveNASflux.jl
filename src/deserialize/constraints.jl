
# Stuff in here should probably move to NaiveNASflux once mature enough...


struct SizePseudoTransparent <: NaiveNASlib.DecoratingTrait
    base::NaiveNASlib.MutationTrait
end
NaiveNASlib.base(t::SizePseudoTransparent) = t.base


NaiveNASlib.all_in_Δsize_graph(::SizePseudoTransparent, d, v, visited) = all_in_Δsize_graph(SizeInvariant(), d, v, visited)

"""
    reshape_keepshape(x, shape)

Same as `reshape` except that `shape[i] = 0` results in `size(y, i+offs) == size(x, i)` where `y` is the output of this function and `offs = ndims(x) - length(shape)`.

This is basically trying to be compliant to the ONNX Reshape operator although the description there of how to interpret a shape of `0` is a bit vague.
"""
function reshape_keepshape(x, shape)
    offs = ndims(x) - length(shape)
    newshape = map(enumerate(shape)) do (ind, new)
        new == 0 && return size(x, ind+offs)
        return new
    end
    return reshape(x, newshape...)
end


"""
    Reshape{T}
    Reshape(dims...; activation_dim=actdim(guess_layertype(length(dims))))
    Reshape(dims; activation_dim=actdim(guess_layertype(length(dims))))

Reshape operation wrapped in a struct for the sole purpose of handling size changes in neighbouring vertices.

Ensures that changes in output size of the input vertex are possible to reshape into the shape given by `dims`.

Will treat `Integer`s in dims as fixed, meaning that if `dims[activation_dim] isa Integer` the input size of the output vertices will be fixed as well.
"""
mutable struct Reshape{T}
    adim::Int
    dims::T
end
Reshape(dims...; activation_dim=actdim(guess_layertype(length(dims)))) = Reshape(activation_dim, dims)
Reshape(dims; activation_dim=actdim(guess_layertype(length(dims)))) = Reshape(activation_dim, dims)

(r::Reshape)(x) = any(==(0), r.dims) ? reshape_keepshape(x, r.dims) : reshape(x, r.dims)

function calc_outsize(r::Reshape, invertex)
    outshape = r.dims[r.adim]
    outshape == 0 && return nout(invertex)
    outshape isa Integer && return outshape
    outshape isa Colon && return 0 # Must be set later...
    error("Unknown outshape: " + outshape)
end


function NaiveNASlib.mutate_inputs(r::Reshape, ins) end
function NaiveNASlib.mutate_outputs(r::Reshape{<:Tuple}, outs)
    r.dims = Tuple(map(enumerate(r.dims)) do (i, s)
        s isa Colon && return s
        i == r.adim && return length(outs)
        return s
    end)
end

NaiveNASlib.minΔninfactor(r::Reshape) = minimum(filter(dim -> dim isa Integer && dim != 0, collect(r.dims)))
NaiveNASlib.minΔnoutfactor(r::Reshape) = minΔninfactor(r)

NaiveNASflux.layer(r::Reshape) = r
NaiveNASflux.actdim(r::Reshape) = r.adim
NaiveNASflux.actrank(r::Reshape) = length(r.dims)

# What about compconstraint for NaiveNASlib.AbstractJuMPSelectionStrategy? Ugh... I give up! Will be treated as SizeAbsorb, i.e no attempt to map elements between input and outputs.

# Special case: Reshape to 2D with variable batch size. Maybe a more general case when this is ok is hiding here somewhere...
NaiveNASlib.compconstraint!(s::NaiveNASlib.AbstractJuMPΔSizeStrategy, r::Reshape{Tuple{Int, Colon}}, data) = reshape_nout_constraint(s, Colon(), r, filter(vin -> vin in keys(data.noutdict), inputs(data.vertex)), data)

function NaiveNASlib.compconstraint!(s::NaiveNASlib.AbstractJuMPΔSizeStrategy, r::Reshape, data)
    ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))

    reshape_nout_constraint(s, r.dims[r.adim], r, ins, data)
    isempty(ins) && return

    fixeddims = filter(dim -> dim isa Integer && dim != 0, collect(r.dims))
    isempty(fixeddims) && return

    if length(fixeddims) == length(r.dims)
        # No size change possible!
        @constraint(data.model, [i=1:length(ins), j=1:length(fixeddims)], data.noutdict[ins[i]] == nout(ins[i]))
        return
    end

    # Make sure that nout of inputs is an integer multiple of at least one of the output sizes
    # It is unfortunately too restrictive to just enforce fv_dims[j] * fixeddims[j] == nout[ins[i]] ∀ i, j
    # as it is enough that this is fulfilled for one j

    # Unfortunately, something like  fv_dims[j] * fixeddims[j] == nout[ins[i]] * b[i] where b is a binary variable is not possible as it is nonlinear.
    # Instead we use i slack(ish) variables with lower bound 0 and force the smallest of them to be 0 (i.e at least one dimension has zero slack).
    fv_dims = @variable(data.model, [1:length(fixeddims)], integer=true)
    slack = @variable(data.model, [1:length(fixeddims)], integer=true, lower_bound=0)
    @constraint(data.model, [i=1:length(ins), j=1:length(fixeddims)], fv_dims[j] * fixeddims[j] ==  data.noutdict[ins[i]] + slack[j])

    # Force the smallest slack to be 0 through a big-M strategy
    atleast_one = @variable(data.model, [1:length(fixeddims)], binary=true)
    min_slack = @variable(data.model, integer=true)
    M = 10000
    @constraint(data.model,[i=1:length(fixeddims)], -M * (1 - atleast_one[i]) <= min_slack - slack[i])
    @constraint(data.model,[i=1:length(fixeddims)], min_slack - slack[i] <= M * (1-atleast_one[i]))
    @constraint(data.model, sum(atleast_one) == 1)
    @constraint(data.model, min_slack == 0)
end


# Case 1: Output size is fixed so we will only put constraints on the inputs to ensure reshaping is possible
function reshape_nout_constraint(s, outsize::Integer, r, ins, data)
    if outsize == 0

        fix_size = true
        for iv in ins
            inrank = unique(actrank(iv))[] # Should be one element or else we are fked
            inadim = unique(actdim(iv))[] # Should be one element or else we are fked

            if inrank - length(r.dims) + r.adim + 1== inadim
                # Dimension to keep size of happens to be input layers activation dimension.
                # This is basically a SizeInvariant vertex
                @constraint(data.model, [i=1:length(ins)], data.noutdict[data.vertex] == data.noutdict[iv])
                fix_size = false
            end
        end

        if fix_size
            @constraint(data.model, [i=1:length(ins)], data.noutdict[data.vertex] == nout(data.vertex))
        end

    else
        @constraint(data.model, data.noutdict[data.vertex] == outsize)
    end
end

# Case 2: Outsize is not fixed so we need to change it so that the output vertices change their input size
function reshape_nout_constraint(s, outsize::Colon, r, ins, data)
    @constraint(data.model, [i=1:length(ins)], data.noutdict[data.vertex] / nout(data.vertex) == data.noutdict[ins[i]] / nout(ins[i]))
end


"""
    Flatten
    Flatten(dim)

Flattens input array along `dim` to a 2D array.

If input has shape `(d_1, d_2, ... d_n)` then the output will have shape `(d_1 X d_2 ... d_(dim-1), d_dim X d_(dim+1) ... X dn)`.

If `dim = 0`, the shape of the output is `(d_1 X d_2 ... d_n, 1)`, where the shape of the input is `(d_0, d_1, ... d_n)`.
"""
struct Flatten
    dim::Int
end

(f::Flatten)(x) = flatten(x, f.dim)
function flatten(x, dim)
    dim == 0 && return reshape(x, :, 1)
    xs = size(x)
    absdim = dim < 0 ? length(xs) + dim : dim
    return reshape(x, prod(xs[1:absdim]), prod(xs[absdim+1:end]))
end

function NaiveNASlib.mutate_inputs(f::Flatten, ins) end
function NaiveNASlib.mutate_outputs(f::Flatten, outs) end

NaiveNASlib.minΔninfactor(f::Flatten) = 1
NaiveNASlib.minΔnoutfactor(f::Flatten) = 1

NaiveNASflux.layer(f::Flatten) = f
NaiveNASflux.actdim(f::Flatten) = 1
NaiveNASflux.actrank(f::Flatten) = 1

calc_outsize(f::Flatten, invertex) = 0 # Must be set later...

function NaiveNASlib.compconstraint!(s::NaiveNASlib.AbstractJuMPΔSizeStrategy, f::Flatten, data)
    ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))
    for iv in ins
        inadim = unique(actdim(iv))[] # Should be one element or else we are fked

        absdim = f.dim > 0 ? f.dim : f.dim + unique(actrank(iv))[] + 1 # Should be one element or else we are fked

        if f.dim == 0 || inadim <= absdim
            # Size of input affect output size
            @constraint(data.model, [i=1:length(ins)], data.noutdict[data.vertex] / nout(data.vertex) ==  data.noutdict[iv] / nout(iv))
        end
        # else: Size of input affects batch size => no constraint needed
    end
end
