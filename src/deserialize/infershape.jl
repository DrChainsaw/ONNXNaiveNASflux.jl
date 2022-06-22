struct MeasureNout{T} <: NaiveNASflux.AbstractMutableComp
    wrapped::T
    actdim::Int
    outsize::Ref{Int}
end
MeasureNout(l; actdim=actdim(l), outsize=0) = MeasureNout(l, actdim, Ref(outsize))
NaiveNASflux.wrapped(mn::MeasureNout) = mn.wrapped
NaiveNASflux.layer(mn::MeasureNout) = layer(mn.wrapped)

NaiveNASlib.nout(mn::MeasureNout) = mn.outsize[]
NaiveNASlib.nin(mn::MeasureNout) = nin(mn.wrapped)

function NaiveNASlib.Δsize!(mn::MeasureNout, ins::AbstractVector, outs::AbstractVector; kwargs...)
    mn.outsize[] = length(outs)
    Δsize!(mn.wrapped, ins, outs; kwargs...)
end

function (mn::MeasureNout)(x...)
    out = mn.wrapped(x...)
    measuresize!(mn, out)
    return out
end
measuresize!(mn::MeasureNout, x::AbstractArray) = mn.outsize[] = size(x, mn.actdim)
function measuresize!(::MeasureNout, x) end # Need methods for AbstractProbe?

calc_outsize(m::NaiveNASflux.AbstractMutableComp, v) = calc_outsize(wrapped(m), v)
function calc_outsize(mn::MeasureNout, v) 
    if mn.outsize[] == 0
        mn.outsize[] = calc_outsize(mn.wrapped, v)
    end
    return mn.outsize[]
end

function try_infer_sizes!(g, insizes...)
    all(v -> nout(v) > 0, vertices(g)) && return
    try
        insizes_nobatch = map(zip(insizes, layertype.(inputs(g)))) do (insize, lt)
            # not sure this can happen...
            lt isa FluxRecurrent && length(insize) == 3 && return insize[1:end-2]
            insize[1:end-1]
        end
        if length(insizes) === length(inputs(g)) && all(inshape -> !isempty(inshape) && all(s -> s isa Number && s > 0, inshape), insizes_nobatch) 
            # This will make any MeasureNout to become aware of the size
            Flux.outputsize(g,insizes_nobatch...; padbatch=true)
            update_size_meta!(g)
        else
            @warn  "No valid input sizes provided. Shape inference could not be done. Either provide Integer insizes manually or use load(...; infer_shapes=false) to disable. If disabled, graph mutation might not work."
        end
        # If insizes are not provided there was an old method called fix_zerosizes! which might be refactored to work with NaiveNASlib 2.0 and MeasureNouts
        # it was not 100% to succeed either and required a CompGraphBuilder
    catch e
        throw(CompositeException([ErrorException("Size inference failed! Use load(...; infer_shapes=false) to disable! If disabled, graph mutation might not work."), e]))
    end
end

update_size_meta!(g::CompGraph) = foreach(update_size_meta!, vertices(g))
update_size_meta!(v::AbstractVertex) = update_size_meta!(base(v))
update_size_meta!(::InputVertex) = nothing
update_size_meta!(v::CompVertex) = update_size_meta!(v.computation)
update_size_meta!(m::NaiveNASflux.AbstractMutableComp, args...) = update_size_meta!(wrapped(m), args...)

function update_size_meta!(m::ActivationContribution, args...)
    outsize = update_size_meta!(wrapped(m), args...)
    if (outsize !== missing) && (m.contribution[] === missing || length(m.contribution[]) != outsize)
        m.contribution[] = zeros(eltype(m.contribution[]), outsize)
    end
    return outsize
end

function update_size_meta!(m::LazyMutable, args...)
    outsize = update_size_meta!(wrapped(m), args...)
    if (outsize !== missing) && (m.outputs === missing || length(m.outputs) != outsize)
        m.outputs = 1:outsize
    end
    return outsize
end

update_size_meta!(m::MeasureNout, args...) = update_size_meta!(wrapped(m), nout(m))
update_size_meta!(f, outsize::Integer) = outsize
update_size_meta!(f) = missing


