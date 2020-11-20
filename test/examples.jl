

@testset "Basic ser/deser example" begin
    using ONNXmutable, Test, Statistics

    l1 = Conv((3,3), 2=>3, relu)
    l2 = Dense(3, 4, elu)

    f = function(x,y)
        x = l1(x)
        # Home-brewed global average pool
        x = dropdims(mean(x, dims=(1,2)), dims=(1,2))
        x = l2(x)
        return x + y
    end

    # Serialize f. Use a string to save to file instead
    io = PipeBuffer()
    x_shape = (:W, :H, 2, :Batch)
    y_shape = (4, :Batch)
    onnx(io, f, x_shape, y_shape)

    # Deserialize as a NaiveNASflux CompGraph (or use the deserialization method from ONNX.jl)
    g = CompGraph(io)

    x = ones(Float32, 5,4,2,3)
    y = ones(Float32, 4, 3)
    @test g(x,y) ≈ f(x,y)

    # Serialization of CompGraphs does not require input shapes to be provided as they can be inferred.
    io = PipeBuffer()
    onnx(io, g)

    g = CompGraph(io)
    @test g(x,y) ≈ f(x,y)

    @test name.(vertices(g)) == ["data_0", "conv_0", "reducemean_0", "squeeze_0", "dense_0", "data_1", "add_0"]
    @test nout.(vertices(g)) == [2, 3, 3, 3, 4, 4, 4]
end

@testset "Check supported OPs list" begin
    # I'm sure there is a better way to do this...
    sosec = "## Supported Operations"
    bts = "```"
    want = sosec
    add = false
    found = String[]
    for line in eachline(joinpath(dirname(pathof(ONNXmutable)),"..", "README.md"))
        add && line == bts && break
        add && !isempty(line) && push!(found, line)
        if line == sosec
            want = bts
        end
        if want == bts && line == bts
            add =true
        end
    end

    io = IOBuffer()
    ONNXmutable.list_supported_ops(io)

    actual = split(String(take!(io)), "\n"; keepempty=false);

    @test found == actual
end

@testset "Custom serialization example" begin

    import ONNXmutable: AbstractProbe, recursename, nextname, newfrom, add!, name, ONNX
    function myfun(probes::AbstractProbe...)
        p = probes[1] # select any probe
        optype = "MyOpType"
        # Naming strategy (e.g. how to avoid duplicate names) is provided by the probe
        # Not strictly needed, but the onnx model is basically corrupt if duplicates exist
        nodename = recursename(optype, nextname(p))

        # Add ONNX node info
        add!(p, ONNX.NodeProto(
        # Names of input is provided by probes. This is why new probes need to be provided as output
        input = collect(name.(probes)),
        # Name of output from this node
        output = [nodename],
        op_type = optype))

        # Probes can procreate like this
        return newfrom(p, nodename, s -> s)
    end

    gp = ONNXmutable.graphproto()
    pps = ONNXmutable.inputprotoprobe!.(Ref(gp), ("in1", "in2"), ((1,2), (3,4)), s -> "out")
    pp = myfun(pps...)

    @test name(pp) == "out"
    @test length(gp.node) == 1
    @test gp.node[1].input == ["in1", "in2"]
    @test gp.node[1].output == ["out"]
    @test gp.node[1].op_type == "MyOpType"

end

@testset "Custom deserialization example" begin
    import ONNXmutable: actfuns

    # All inputs which are not output from another node in the graph are provided in the method call
    actfuns[:MyActFun1] = (params, α, β) -> x -> x^α + β
    # Params contains a Dict with attributes.
    actfuns[:MyActFun2] = function(params)
        α = get(params, :alpha, 1)
        return x -> α / x
    end
    ONNXmutable.refresh()

    af1 = ONNXmutable.verts[:MyActFun1]("af1", [inputvertex("in", 3)], Dict(), 2, 3)
    @test af1(4) == 4^2 + 3

    af2 = ONNXmutable.verts[:MyActFun2]("af2", [inputvertex("in", 3)], Dict(:alpha => 3))
    @test af2(4) == 3/4

    delete!(ONNXmutable.verts, :MyActFun1)
    delete!(ONNXmutable.verts, :MyActFun2)
    delete!(ONNXmutable.invariantops, :MyActFun1)
    delete!(ONNXmutable.invariantops, :MyActFun2)
    delete!(actfuns, :MyActFun1)
    delete!(actfuns, :MyActFun2)
    ONNXmutable.refresh()
end
