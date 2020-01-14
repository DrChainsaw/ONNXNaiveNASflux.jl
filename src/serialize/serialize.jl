
function default_namestrat(vs::AbstractVector{<:AbstractVertex})
    all(isnamed, vs) && return v -> name(v)
    namegen = name_runningnr()
    ng(v::AbstractVertex) = namegen
    ng(f) = namegen(f)
    return ng
end

isnamed(v::AbstractVertex) = isnamed(base(v))
isnamed(v::CompVertex) = false
isnamed(v::InputVertex) = true
isnamed(v::MutationVertex) = isnamed(trait(v))
isnamed(t::DecoratingTrait) = isnamed(base(t))
isnamed(t) = false
isnamed(::NamedTrait) = true

function name_runningnr(namefun = genname)
    exists = Set{String}()

    return function(f)
        bname = genname(f)
        next = 0
        candname = bname * "_" * string(next)
        while candname in exists
            next += 1
            candname = bname * "_" * string(next)
        end
        push!(exists, candname)
        return candname
    end
end

#genname(f::F{T...}) where {F,T...} = string(F)
genname(f::F) where F = lowercase(string(F.name))
genname(s::AbstractString) = s
genname(f::Function) = lowercase(string(f))

function NaiveNASlib.vertices(outs::AbstractVector{<:AbstractVertex}, ins::AbstractVector{<:AbstractVertex}=[])
    verts = copy!(AbstractVertex[], ins)
    foreach(vo -> flatten(vo, verts), outs)
    return verts
end

graphproto(g::CompGraph, outshapes = v -> (nout(v), layertype(v)), namestrat=default_namestrat(vertices(g))) = graphproto(g.outputs, g.inputs; outshapes = outshapes, namestrat=namestrat)


function graphproto(outs::AbstractVector{<:AbstractVertex}, ins::AbstractVector{<:AbstractVertex} = []; outshapes = v -> (nout(v), layertype(v)), namestrat=default_namestrat(vertices(outs, ins)))

    gp = ONNX.Proto.GraphProto(
    node = ONNX.Proto.NodeProto[],
    initializer =  ONNX.Proto.TensorProto[],
    input =  ONNX.Proto.ValueInfoProto[],
    output =  ONNX.Proto.ValueInfoProto[],
    value_info =  ONNX.Proto.ValueInfoProto[]
    )

    outnames = IdDict()
    for v in vertices(outs, ins)
        innames = map(vi -> outnames[vi], inputs(v))
        outnames[v] = add!(gp, v, outshapes, innames, namestrat)
    end

    gp.output = map(outs) do vo
        ONNX.Proto.ValueInfoProto(outnames[vo], outshapes(vo))
    end

    return gp
end

add!(gp, v::NaiveNASflux.InputShapeVertex, args...) = add_inputvertex(gp, v, args...)
add!(gp, v::InputSizeVertex, args...) = add_inputvertex(gp, v, args...)
add!(gp, v::InputVertex, args...) = add_inputvertex(gp, v, args...)

add_inputvertex(gp, v, outshapes, innames, namestrat) = add_inputvertex(gp, v, outshapes, namestrat(v))
add_inputvertex(gp, v, outshapes, namestrat::Function) = add_inputvertex(gp, v, outshapes, namestrat(name(v)))

function add_inputvertex(gp, v, outshapes, vname::AbstractString)
    push!(gp.input, ONNX.Proto.ValueInfoProto(vname, outshapes(v)))
    return vname
end

function add!(gp, v::AbstractVertex, outshapes, innames, namestrat)
    foreach(p -> add!(gp, p), protos(v, innames, namestrat))
    return gp.node[end].name
end

add!(gp, np::ONNX.Proto.NodeProto) = push!(gp.node, np)

function add!(gp, tp::ONNX.Proto.TensorProto)
    push!(gp.initializer, tp)
    push!(gp.input, ONNX.Proto.ValueInfoProto(tp.name, tp.dims))
end

# Onion peeling...
protos(v::MutationVertex, innames, namestrat::Function) = protos(base(v), innames, namestrat(v))
protos(v::AbstractVertex, innames, namestrat) = protos(base(v), innames, namestrat)
protos(v::InputVertex, innames, namestrat) = ()
protos(v::CompVertex, innames, namestrat) = protos(v.computation, innames, namestrat)

protos(m::AbstractMutableComp, innames, namestrat) = protos(NaiveNASflux.wrapped(m), innames, namestrat)

# Why why why?!?! Because namestrat(v) is allowed to return either a string or a function translating a layer/function to a string, all for the convenience of the user
protos(l, innames, namestrat) = protos(l, innames, namestrat(l))

function protos(l::Dense, innames, lname::AbstractString)
    @assert length(innames) == 1 "Only one input expected! Got $innames"
    wname, bname = lname .* ("_weight", "_bias")

    return ONNX.Proto.NodeProto(
        input=[innames[], wname, bname],
        output=[lname],
        name=lname,
        op_type="Gemm"),
        ONNX.Proto.TensorProto(weights(l), wname),
        ONNX.Proto.TensorProto(bias(l), bname),
        protos(l.σ, [lname], f -> join([lname, lowercase(string(f))], "_"))...
end

protos(::typeof(identity), innames, name::AbstractString) = ()

function protos(::typeof(relu), innames, lname::AbstractString)
    return (ONNX.Proto.NodeProto(
    input = innames,
    output = [lname],
    name=lname,
    op_type="Relu"), )
end
