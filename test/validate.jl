@testset "Validate" begin
    import ONNXNaiveNASflux: modelproto, graphproto
    import ONNXNaiveNASflux: validate, uniqueoutput, optypedefined, outputused, inputused, hasname, ONNX.NodeProto

    np(ins,outs,op="A") = ONNX.NodeProto(input=ins, output=outs, op_type=op)
    vip(name,shape=(2,:A)) = ONNX.ValueInfoProto(name, shape)

    @testset "Duplicate output name" begin
        nodes = [np(["in"], ["n1"])]
        push!(nodes, np(["n1"], ["n2"]))
        push!(nodes, np(["n2"], ["n1", "n3"]))

        gp = graphproto(;node=nodes)
        mp = modelproto(;graph=gp)
        @test_throws ErrorException uniqueoutput(mp)
        @test_throws ErrorException validate(mp)
        @test_logs (:warn, r"Duplicate output name: n1 found in NodeProto.*in.*n1.* and NodeProto.*n2.*n1.*n3") uniqueoutput(mp, s -> @warn replace(s, "\n" => ""))
    end

    @testset "No OP specified" begin
        nodes = [ONNX.NodeProto(output=["n1"], input=["in"])]

        gp = graphproto(;node=nodes)
        mp = modelproto(;graph=gp)
        @test_throws ErrorException optypedefined(mp)
        @test_throws ErrorException validate(mp)
        @test_logs (:warn, r"No op_type defined.*in.*n1") optypedefined(mp, s -> @warn replace(s, "\n" => ""))
    end

    @testset "All outputs not used" begin
        nodes = [np(["in"], ["n1", "nf1"])]
        push!(nodes, np(["n1"], ["n2", "nf2"]))
        push!(nodes, np(["n2"], ["n3"]))
        output = vip.(["n3", "n1"])
        input = vip.(["in", "nf3"])

        gp = graphproto(;node=nodes, input=input, output=output)
        mp = modelproto(;graph=gp)
        @test_throws ErrorException outputused(mp)
        @test_throws ErrorException validate(mp)
        @test_logs (:warn, "Found unused outputs: nf1, nf2, nf3") outputused(mp, s -> @warn s)
    end

    @testset "All inputs not used" begin
        nodes = [np(["in"], ["n1"])]
        push!(nodes, np(["n1", "nf1"], ["n2"]))
        push!(nodes, np(["n2"], ["n3"]))
        output = vip.(["n3", "n1", "nf2"])
        input = vip.(["in"])

        gp = graphproto(;node=nodes, input=input, output=output)
        mp = modelproto(;graph=gp)
        @test_throws ErrorException inputused(mp)
        @test_throws ErrorException validate(mp)
        @test_logs (:warn, "Found unused inputs: nf1, nf2") inputused(mp, s -> @warn s)
    end

    @testset "Graph has a name" begin
        gp = ONNX.GraphProto(;node=ONNX.NodeProto[],input=ONNX.ValueInfoProto[], output=ONNX.ValueInfoProto[],initializer=ONNX.TensorProto[],)
        
        mp = modelproto(;graph=gp)
        @test_throws ErrorException hasname(mp)
        @test_throws ErrorException validate(mp)
        @test_logs (:warn, "Graph name is empty string!") hasname(mp, s -> @warn s)
    end
end
