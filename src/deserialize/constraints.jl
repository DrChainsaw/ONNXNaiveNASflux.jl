

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
    offs = max(0, ndims(x) - length(shape))
    newshape = map(enumerate(shape)) do (ind, new)
        new == 0 && return size(x, ind+offs)
        return new
    end
    return reshape(x, newshape...)
end


"""
    Reshape{T}
    Reshape(dims...; activation_dim=actdim(length(dims)))
    Reshape(dims; activation_dim=actdim(length(dims)))

Reshape operation wrapped in a struct for the sole purpose of handling size changes in neighbouring vertices.

Ensures that changes in output size of the input vertex are possible to reshape into the shape given by `dims`.

Will treat `Integer`s in dims as fixed, meaning that if `dims[activation_dim] isa Integer` the input size of the output vertices will be fixed as well.
"""
struct Reshape{T}
    adim::Int
    dims::T
end
Reshape(dims...; activation_dim=actdim(length(dims))) = Reshape(activation_dim, dims)
Reshape(dims; activation_dim=actdim(length(dims))) = Reshape(activation_dim, dims)

(r::Reshape)(x) = any(==(0), r.dims) ? reshape_keepshape(x, r.dims) : reshape(x, r.dims)

function NaiveNASlib.mutate_inputs(r::Reshape, ins) end
function NaiveNASlib.mutate_outputs(r::Reshape, outs) end

# What about compconstraint for NaiveNASlib.AbstractJuMPSelectionStrategy? Ugh... I give up! Should be treated as SizeAbsorb, i.e no attempt to map elements between input and outputs.

function NaiveNASlib.compconstraint!(s::NaiveNASlib.AbstractJuMPΔSizeStrategy, r::Reshape, data)
    ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))

    reshape_nout_constraint(s, r.dims[r.adim], ins, data)
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
function reshape_nout_constraint(s, outsize::Integer, ins, data)
    if outsize == 0
        @constraint(data.model, [i=1:length(ins)], data.noutdict[data.vertex] == nout(data.vertex))
    else
        @constraint(data.model, data.noutdict[data.vertex] == outsize)
    end
end

# Case 2: Outsize is not fixed so we need to change it so that the output vertices change their input size
function reshape_nout_constraint(s, outsize::Colon, ins, data)
    @constraint(data.model, [i=1:length(ins)], data.noutdict[data.vertex] / nout(data.vertex) == data.noutdict[ins[i]] / nout(ins[i]))
end
