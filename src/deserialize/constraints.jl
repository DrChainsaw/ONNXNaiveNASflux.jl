

struct Reshape{T}
    dims::T
end
(r::Reshape)(x) = reshape(x, r.dims)
function NaiveNASlib.mutate_inputs(r::Reshape, ins) end
function NaiveNASlib.mutate_outputs(r::Reshape, outs) end

NaiveNASlib.compconstraint!(s::NaiveNASlib.AbstractJuMPΔSizeStrategy, r::Reshape, data) = reshape_constraint(s, r.dims[actdim(length(r.dims))], r, data)


# Case 1: Output size is fixed so we will only put constraints on the inputs to ensure reshaping is possible
function reshape_constraint(s, outsize::Integer, r, data)
    ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))

    @constraint(data.model, data.noutdict[data.vertex] == outsize)
    isempty(ins) && return

    fixeddims = filter(dim -> dim isa Integer, collect(r.dims))

    # TODO: What if there is no colon? Will the below work then or do we need to freeze the sizes?

    # Make sure that nout of inputs is an integer multiple of at least one of the output sizes
    # It is unfortunately too restrictive to just enforce fv_dims[j] * fixeddims[j] == nout[ins[i]] ∀ i, j
    # as it is enough that this is fulfilled for one j

    # Unfortunately, something like  fv_dims[j] * fixeddims[j] == nout[ins[i]] * b[i] where b is a binary variable is not possible as it is nonlinear.
    # Instead we use i slack(ish) variables with lower bound 0 and force the smallest of them to be 0 (i.e at least one dimension has zero slack).
    fv_dims = @variable(data.model, [1:length(fixeddims)], integer=true)
    slack = @variable(data.model, [1:length(fixeddims)], integer=true, lower_bound=0)
    @constraint(data.model, [i=1:length(ins), j=1:length(fixeddims)], fv_dims[j] * fixeddims[j] ==  data.noutdict[ins[i]] + slack[j])

    # Force the minimum of slack to be 0 through a big-M strategy
    atleast_one = @variable(data.model, [1:length(fixeddims)], binary=true)
    min_slack = @variable(data.model, integer=true)
    M = 10000
    @constraint(data.model,[i=1:length(fixeddims)], -M * (1 - atleast_one[i]) <= min_slack - slack[i])
    @constraint(data.model,[i=1:length(fixeddims)], min_slack - slack[i] <= M * (1-atleast_one[i]))
    @constraint(data.model, sum(atleast_one) == 1)
    @constraint(data.model, min_slack == 0)

end
