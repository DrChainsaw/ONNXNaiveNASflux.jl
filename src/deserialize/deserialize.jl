

"""
   extract(modelfile)

Return a [`ONNX.Types.Model`](@ref) and a Dict mapping input variables to size tuples (in Flux order).

Beware that missing/variable size data for a dimension results in a random size for that dimension. Therefore sizes should mostly be used to determine the number of dimensions.
"""
extract(modelfile::AbstractString) = open(io -> extract(io), modelfile)
extract(io::IO) = ONNX.readproto(io, ONNX.ModelProto())

sizes(mp::ONNX.ModelProto) = sizes(mp.graph)
sizes(gp::ONNX.GraphProto) = Dict((name.(gp.input) .=> size.(gp.input))..., (name.(gp.output) .=> size.(gp.output))...)

NaiveNASlib.name(n::OnnxNode) = name(n.proto)
NaiveNASlib.name(n::ONNX.NodeProto) = n.name
NaiveNASlib.name(vip::ONNX.ValueInfoProto) = vip.name
NaiveNASlib.name(tp::ONNX.TensorProto) = tp.name

"""
   CompGraph(filename::String)

Return a [`CompGraph`](@ref) loaded from the given file.
"""
NaiveNASlib.CompGraph(filename::String, vfun = create_vertex_default) = open(io -> CompGraph(io, vfun), filename)
NaiveNASlib.CompGraph(io::IO, vfun = create_vertex_default) = CompGraph(extract(io), vfun)
NaiveNASlib.CompGraph(m::ONNX.ModelProto, vfun = create_vertex_default) = CompGraph(m.graph, vfun)

function NaiveNASlib.CompGraph(g::ONNX.GraphProto, vfun = create_vertex_default)
   gb = CompGraphBuilder(g)
   outputs::Vector{AbstractVertex} = vertex.(gb, node.(name.(g.output), gb), vfun)
   graph = CompGraph(gb.inputs, outputs)
   fix_zerosizes!.(outputs, gb)
   return graph
end

# This is pretty specific to NaiveNASlib as it has size metadata for all vertices to aid in lazy mutations
# Sometimes the size values are missing from the protos and then we do this little thing to try to infer it from other sources under the assumption that the imported graph is size-consistent
fix_zerosizes!(v::AbstractVertex, gb) = fix_zerosizes!(base(v), gb)
function fix_zerosizes!(::InputVertex, ::Any) end
function fix_zerosizes!(::CompVertex, ::Any) end
function fix_zerosizes!(::SourceVertex, ::Any) end
function fix_zerosizes!(v::MutationVertex, gb)
    
    if nout(v) == 0
        outs = outputs(v)
        if !isempty(outs)
            vo = first(outs)
            ind = findfirst(==(v), inputs(vo))
            startnout = nin(vo)[ind]
            Δnout(op(v), startnout)
            NaiveNASlib.reset_out!(op(v))
        elseif name(v) in keys(gb.sizes)
            # Beware! Uninitialized sizes result in random sizes when loaded?!?!
            # Lets avoid too big sizes
            startnout = gb.sizes[name(v)][first(actdim(v))]
            if startnout < 1e8
                Δnout(op(v), startnout)
                NaiveNASlib.reset_out!(op(v))
            end
        end
    end

    for (ind, curr_insize) in enumerate(nin(v))
        found_insize = findinsize(layertype(v), v, ind, gb)
        if curr_insize == 0 && found_insize != 0
            toset = zeros(Int, length(nin(v)))
            toset[ind] = found_insize
            Δnin(op(v), toset...)
            NaiveNASlib.reset_in!(op(v))
        elseif curr_insize != found_insize
            @warn "Mismatched input sizes found for vertex with name $(name(v)) and layertype $(layertype(v)): $curr_insize vs $(found_insize)! Graph mutation near this vertex might not work"
        end
        fix_zerosizes!(inputs(v)[ind], gb)
    end
end

function findinsize(lt, v, in_index, gb)
    insize = nout(inputs(v)[in_index])
    insize != 0 && return insize


    fix_zerosizes!(inputs(v)[in_index], gb)
    return nout(inputs(v)[in_index])
end
findinsize(::FluxParLayer, v, in_index, gb) = nin(layer(v))


"""
   vertex(gb::CompGraphBuilder, n::ONNX.Types.Node, vfun = create_vertex_default)

Return an `AbstractVertex` created from `n`.

Inputs to the returned vertex are created recursively based on state in `gb`.
"""
function vertex(gb::CompGraphBuilder, n::OnnxNode, vfun = create_vertex_default)
      return get!(gb.created, n) do
         n_create, ins = check_combine(gb, n)
         invertices = map(ni -> vertex(gb, ni, vfun), ins)
         v = vfun(n_create, invertices)
         if is_input(v)
            push!(gb.inputs, v)
         end
         return v
      end
end

is_input(v::AbstractVertex) = is_input(base(v))
is_input(v::InputVertex) = true
is_input(v::CompVertex) = false
is_input(v::SourceVertex) = false


create_vertex_default(n::OnnxNode, inputs::Array; kwargs...) = verts[optype(n)](name(n), inputs, n.attribute, params(n)...; kwargs...)
