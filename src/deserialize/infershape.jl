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

function calc_outsize(mn, v) 
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
        if length(insizes) === length(inputs(g)) && all(inshape -> all(s -> s isa Number && s > 0, inshape), insizes_nobatch) 
            # This will make any MeasureNout to become aware of the size
            Flux.outputsize(g,insizes_nobatch...; padbatch=true)
        else
            @warn  "No valid input sizes provided. Shape inference could not be done. Either provide Integer insizes manually or use load(...; infer_shapes=false) to disable. If disabled, graph mutation might not work."
        end
        # If insizes are not provided there was an old method called fix_zerosizes! which might be refactored to work with NaiveNASlib 2.0 and MeasureNouts
        # it was not 100% to succeed either and required a CompGraphBuilder
    catch e
        throw(CompositeException([ErrorException("Size inference failed! Use load(...; infer_shapes=false) to disable! If disabled, graph mutation might not work."), e]))
    end
end