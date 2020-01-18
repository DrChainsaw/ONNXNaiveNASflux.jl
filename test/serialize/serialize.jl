
@testset "Structure" begin
    import ONNXmutable: optype, actfuns, fluxlayers
    using NaiveNASflux
    import NaiveNASflux: weights, bias

    function serdeser(p::T, convfun = cfun(p)) where T
        iob = PipeBuffer();
        ONNX.writeproto(iob, p)
        return convfun(ONNX.readproto(iob, T()))
    end

    cfun(pt) = ONNX.convert
    cfun(::ONNX.Proto.TensorProto) = ONNX.get_array
    cfun(::ONNX.Proto.GraphProto) = gp -> (ONNX.convert(gp), ONNXmutable.sizes(gp))

    @testset "Nodes" begin
        import ONNXmutable: AbstractProbe, nextname, newfrom, add!, genname
        struct NodeProbe{F} <: AbstractProbe
            name::String
            namefun::F
            protos::Vector{Any}
        end
        NodeProbe(name, namefun) = NodeProbe(name, namefun, [])
        ONNXmutable.add!(p::NodeProbe, n) = push!(p.protos, n)
        ONNXmutable.nextname(p::NodeProbe) = p.namefun
        ONNXmutable.newfrom(p::NodeProbe, pname) = NodeProbe(pname, p.namefun, p.protos)
        ONNXmutable.newnamestrat(p::NodeProbe, f, pname = name(p)) = NodeProbe(pname, f, p.protos)
        ONNXmutable.name(p::NodeProbe) = p.name

        @testset "Paramless function $(tc.f)" for tc in (
            (f=relu, ot="Relu")
            ,)

            inprobe = NodeProbe("input", f -> "output")

            outprobe = tc.f(inprobe)

            @test length(outprobe.protos) == 1

            res = serdeser(outprobe.protos[])

            @test res.input == [name(inprobe)]
            @test res.output == [name(outprobe)]
            @test res.op_type == tc.ot
            @test res.name == name(outprobe)
        end

        @testset "$(tc.layer) node" for tc in (
            (layer=Dense(3,4, relu), indata=reshape(collect(1:12), :, 4) .- 3),
            (layer=Conv((1,2), 3=>4, relu), indata=reshape(collect(1:2*3*3), 2,3,3,1) .- 10)
            )

            inprobe = NodeProbe("input", genname)

            outprobe = tc.layer(inprobe)

            @test length(outprobe.protos) == 4

            dp,wp,bp,ap = Tuple(outprobe.protos)

            dn = serdeser(dp)
            an = serdeser(ap)
            w = serdeser(wp)
            b = serdeser(bp)

            @test size(w) == size(weights(tc.layer))
            @test size(b) == size(bias(tc.layer))

            @test w ≈ weights(tc.layer)
            @test b ≈ bias(tc.layer)

            dn.attribute[:activation] = actfuns[Symbol(optype(an))](an.attribute)
            res = fluxlayers[optype(dn)](dn.attribute, w, b)

            @test string(res) == string(tc.layer)

            @test res(tc.indata) ≈ tc.layer(tc.indata)
        end
    end

    @testset "Graphs" begin
        import ONNXmutable: graphproto

        dense(name, inpt::AbstractVertex, outsize, actfun=identity) = mutable(name, Dense(nout(inpt), outsize, actfun), inpt)
        dense(inpt::AbstractVertex, outsize, actfun=identity) = mutable(Dense(nout(inpt), outsize, actfun), inpt)

        convvertex(name, inpt::AbstractVertex, outsize, actfun=identity) = mutable(name, Conv((1,1), nout(inpt) => outsize, actfun), inpt)

        function test_named_graph(g_org, extradims = ())
            gp_org = graphproto(g_org)
            gt_new, sizes = serdeser(gp_org)

            g_new = CompGraph(gt_new, sizes)

            @test name.(vertices(g_org)) == name.(vertices(g_new))

            outsize = nout(g_org.inputs[1])
            bs = 4
            indata = reshape(collect(Float32, 1:outsize*bs*prod(extradims)), extradims..., outsize, :)
            @test g_org(indata) ≈ g_new(indata)
            return g_new
        end

        @testset "Linear Dense graph with names" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense("dense1", v0, 4, relu)
            v2 = dense("dense2", v1, 5, relu)
            v3 = dense("output", v2, 2)

            test_named_graph(CompGraph(v0, v3))
        end

        @testset "Linear Dense graph without names" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense(v0, 4, relu)
            v2 = dense(v1, 5, relu)
            v3 = dense(v2, 2)

            g_org = CompGraph(v0, v3)

            gp_org = graphproto(g_org)
            gt_new, sizes = serdeser(gp_org)

            g_new = CompGraph(gt_new, sizes)

            @test name.(vertices(g_new)) == ["input_0", "dense_0", "dense_1", "dense_2"]

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "Linear Conv graph with names" begin
            v0 = inputvertex("input", 3, FluxConv{2}())
            v1 = convvertex("conv1", v0, 4, relu)
            v2 = convvertex("conv2", v1, 5, relu)
            v3 = convvertex("output", v2, 2)

            test_named_graph(CompGraph(v0, v3), (2,3))
        end

        @testset "Dense graph with add" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense("dense1", v0, 4, relu)
            v2 = dense("dense2", v0, 4)
            v3= "add" >> v1 + v2
            v4 = dense("output", v3, 2, relu)

            test_named_graph(CompGraph(v0, v4))
        end

        @testset "Dense graph with add without names" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense(v0, 4, relu)
            v2 = dense(v0, 4)
            v3 = v1 + v2

            g_org = CompGraph(v0, v3)

            gp_org = graphproto(g_org)
            gt_new, sizes = serdeser(gp_org)

            g_new = CompGraph(gt_new, sizes)

            @test name.(vertices(g_new)) == ["input_0", "dense_0", "dense_1", "add_0"]

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "Dense graph with cat" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense("dense1", v0, 4, relu)
            v2 = dense("dense2", v0, 4)
            v3 = concat("conc", v1, v2)
            v4 = dense("output", v3, 2, relu)

            test_named_graph(CompGraph(v0, v4))
        end

        @testset "Dense graph with cat without names" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense(v0, 4, relu)
            v2 = dense(v0, 4)
            v3 = concat("conc", v1, v2)

            g_org = CompGraph(v0, v3)

            gp_org = graphproto(g_org)
            gt_new, sizes = serdeser(gp_org)

            g_new = CompGraph(gt_new, sizes)

            @test name.(vertices(g_new)) == ["input_0", "dense_0", "dense_1", "concat_0"]

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "Graph two inputs two outputs" begin
            vins = inputvertex.(["in1", "in2"], 3, Ref(FluxDense()))
            v1 = "add" >> vins[1] + vins[2]
            v2 = concat("conc", vins[1], vins[2])

            g_org = CompGraph(vins, [v1, v2])
            g_new = CompGraph(serdeser(graphproto(g_org))...)

            @test name.(vertices(g_org)) == name.(vertices(g_new))

            indata1 = reshape(collect(Float32, 1:3*4), nout(vins[1]), :)
            indata2 = indata1 .* -0.5
            @test g_org(indata1, indata2) == g_new(indata1, indata2)
        end
    end
end
