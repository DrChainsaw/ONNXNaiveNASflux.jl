
@testset "Structure" begin
    import ONNXmutable: protos, optype, actfuns, fluxlayers
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

    @testset "Paramless function $(tc.f)" for tc in (
        (f=relu, ot="Relu")
        ,)

        inname = ["input"]
        outname = "output"

        np = protos(tc.f, inname, t -> outname)[1]
        res = serdeser(np)

        @test res.input == inname
        @test res.output == [outname]
        @test res.op_type == tc.ot
        @test res.name == outname
    end

    @testset "Dense layer actfun $af" for af in (
        relu,
        )
        exp = Dense(3,4, af)

        inname = ["input"]

        dp,wp,bp,ap = protos(exp, inname, l -> lowercase(string(typeof(l))))

        dn = serdeser(dp)
        an = serdeser(ap)
        w = serdeser(wp)
        b = serdeser(bp)

        @test size(w) == size(weights(exp))
        @test size(b) == size(bias(exp))

        @test w ≈ weights(exp)
        @test b ≈ bias(exp)

        dn.attribute[:activation] = actfuns[Symbol(optype(an))](an.attribute)
        res = fluxlayers[optype(dn)](dn.attribute, w, b)

        @test string(res) == string(exp)

        indata = reshape(collect(1:4*nin(res)), :, 4) .- 3
        @test res(indata) ≈ exp(indata)
    end

    @testset "Graphs" begin
        import ONNXmutable: graphproto

        dense(name, inpt::AbstractVertex, outsize, actfun=identity) = mutable(name, Dense(nout(inpt), outsize, actfun), inpt)
        dense(inpt::AbstractVertex, outsize, actfun=identity) = mutable(Dense(nout(inpt), outsize, actfun), inpt)

        @testset "Linear Dense graph with names" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense("dense1", v0, 4, relu)
            v2 = dense("dense2", v1, 5, relu)
            v3 = dense("output", v2, 2)

            g_org = CompGraph(v0, v3)

            gp_org = graphproto(g_org)
            gt_new, sizes = serdeser(gp_org)

            g_new = CompGraph(gt_new, sizes)

            @test name.(vertices(g_org)) == name.(vertices(g_new))

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            @test g_org(indata) ≈ g_new(indata)
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
    end
end
