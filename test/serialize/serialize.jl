
@testset "Structure" begin
    import ONNXmutable: optype, actfuns, fluxlayers, invariantops
    using NaiveNASflux
    import NaiveNASflux: weights, bias
    using Statistics

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

        @testset "InvariantOp $(tc.op) attrs: $(pairs(tc.attr))" for tc in (
            (op=:Relu, attr = Dict()),
            (op=:Elu, attr = Dict()),
            (op=:Elu, attr = Dict(:alpha => 5f-1)),
            (op=:Selu, attr = Dict()),
            (op=:Selu, attr = Dict(:alpha => 15f-1)),
            (op=:GlobalAveragePool, attr=Dict()),
            )

            inprobe = NodeProbe("input", f -> "output")

            outprobe = invariantops[tc.op](tc.attr)(inprobe)

            @test length(outprobe.protos) == 1

            res = serdeser(outprobe.protos[])

            @test res.input == [name(inprobe)]
            @test res.output == [name(outprobe)]
            @test res.op_type == string(tc.op)
            @test res.name == name(outprobe)
        end

        @testset "Dims method $(tc.ot)" for tc in (
            (f=cat, dims=1, ndims=2, ot="Concat", axname=:axis),
            (f=mean, dims=(2, 3), ndims=4, ot="ReduceMean", axname=:axes),
            (f=dropdims, dims=(3,), ndims=3, ot="Squeeze", axname=:axes)
            )
            inprobe = NodeProbe("input", f -> "output")
            ONNXmutable.shape(p::NodeProbe) = Tuple(1:tc.ndims)

            outprobe = tc.f(inprobe, dims=tc.dims)

            @test length(outprobe.protos) == 1

            res = serdeser(outprobe.protos[])

            @test res.input == [name(inprobe)]
            @test res.output == [name(outprobe)]
            @test res.op_type == tc.ot
            @test res.name == name(outprobe)
            expdims = tc.dims isa Tuple ? collect(tc.dims) : tc.dims
            @test ONNXmutable.numpy2fluxdim.(res.attribute[tc.axname], tc.ndims) == expdims
        end

        @testset "$(tc.layer) node" for tc in (
            (layer=Dense(3,4, relu), indata=reshape(collect(1:12), :, 4) .- 3),
            (layer=Conv((1,2), 3=>4, relu), indata=reshape(collect(1:2*3*3), 2,3,3,1) .- 10),
            )

            inprobe = NodeProbe("input", genname)

            outprobe = tc.layer(inprobe)

            @test length(outprobe.protos) == 4

            lp,wp,bp,ap = Tuple(outprobe.protos)

            ln = serdeser(lp)
            an = serdeser(ap)
            w = serdeser(wp)
            b = serdeser(bp)

            @test size(w) == size(weights(tc.layer))
            @test size(b) == size(bias(tc.layer))

            @test w ≈ weights(tc.layer)
            @test b ≈ bias(tc.layer)

            ln.attribute[:activation] = actfuns[Symbol(optype(an))](an.attribute)
            res = fluxlayers[optype(ln)](ln.attribute, w, b)

            @test string(res) == string(tc.layer)

            @test res(tc.indata) ≈ tc.layer(tc.indata)
        end

        @testset "$(tc.layer) node" for tc in (
            (layer=BatchNorm(3, relu; initβ = i -> collect(Float32, 1:i), initγ = i -> collect(Float32, i:-1:1), ϵ=1e-3, momentum = 0.78), indata=reshape(collect(1:2*3*3), 2,3,3,1) .- 10),
            )

            inprobe = NodeProbe("input", genname)
            outprobe = tc.layer(inprobe)
            @test length(outprobe.protos) == 6

            ln, γ, β, μ, σ², an = Tuple(serdeser.(outprobe.protos))

            @test size(β) == size(tc.layer.β)
            @test size(γ) == size(tc.layer.γ)
            @test size(μ) == size(tc.layer.μ)
            @test size(σ²) == size(tc.layer.σ²)

            @test β ≈ tc.layer.β
            @test γ ≈ tc.layer.γ
            @test μ ≈ tc.layer.μ
            @test σ² ≈ tc.layer.σ²

            ln.attribute[:activation] = actfuns[Symbol(optype(an))](an.attribute)
            res = fluxlayers[optype(ln)](ln.attribute, γ, β, μ, σ²)

            @test string(res) == string(tc.layer)

            @test res(tc.indata) ≈ tc.layer(tc.indata)
        end
    end

    @testset "Graphs" begin
        import ONNXmutable: graphproto

        dense(name, inpt::AbstractVertex, outsize, actfun=identity) = mutable(name, Dense(nout(inpt), outsize, actfun), inpt)
        dense(inpt::AbstractVertex, outsize, actfun=identity) = mutable(Dense(nout(inpt), outsize, actfun), inpt)

        convvertex(name, inpt::AbstractVertex, outsize, actfun=identity) = mutable(name, Conv((1,1), nout(inpt) => outsize, actfun), inpt)

        bnvertex(name, inpt::AbstractVertex, actfun=identity) = mutable(name, BatchNorm(nout(inpt), actfun), inpt)

        fvertex(name, inpt::AbstractVertex, f) = invariantvertex(f, inpt; traitdecoration = t -> NamedTrait(t, name))

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

        @testset "Linear Dense graph" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense("dense1", v0, 4, relu)
            v2 = dense("dense2", v1, 5, elu)
            v3 = dense("output", v2, 2)

            test_named_graph(CompGraph(v0, v3))
        end

        @testset "Linear Dense graph without names" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense(v0, 4, selu)
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

        @testset "Linear Conv graph" begin
            v0 = inputvertex("input", 3, FluxConv{2}())
            v1 = convvertex("conv1", v0, 4, selu)
            v2 = convvertex("conv2", v1, 5, elu)
            v3 = convvertex("output", v2, 2)

            test_named_graph(CompGraph(v0, v3), (2,3))
        end

        @testset "Linear Conv graph with global pooling" begin
            v0 = inputvertex("input", 3, FluxConv{2}())
            v1 = convvertex("conv1", v0, 4, relu)
            v2 = bnvertex("batchnorm1", v1, elu)
            v3 = fvertex("globmeanpool", v2, x -> ONNXmutable.globalmeanpool(x, y -> dropdims(y, dims=(1,2))))
            v4 = dense("output", v3, 2, selu)

            test_named_graph(CompGraph(v0, v4), (2,3))
        end

        @testset "Linear Batchnorm and Conv graph with global pooling" begin
            v0 = inputvertex("input", 3, FluxConv{2}())
            v1 = convvertex("conv1", v0, 4, relu)
            v2 = convvertex("conv2", v1, 5, elu)
            v3 = fvertex("globmeanpool", v2, x -> ONNXmutable.globalmeanpool(x, y -> dropdims(y, dims=(1,2))))
            v4 = dense("output", v3, 2)

            test_named_graph(CompGraph(v0, v4), (2,3))
        end

        @testset "Dense graph with add" begin
            v0 = inputvertex("input", 3, FluxDense())
            v1 = dense("dense1", v0, 4, relu)
            v2 = dense("dense2", v0, 4)
            v3= "add" >> v1 + v2
            v4 = dense("output", v3, 2, selu)

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
            v1 = dense("dense1", v0, 4, elu)
            v2 = dense("dense2", v0, 4)
            v3 = concat("conc", v1, v2)
            v4 = dense("output", v3, 2, relu)

            test_named_graph(CompGraph(v0, v4))
        end

        @testset "Conv and batchnorm graph with cat" begin
            v0 = inputvertex("input", 3, FluxConv{2}())
            v1 = convvertex("conv", v0, 2, elu)
            v2 = bnvertex("batchnorm", v0)
            v3 = concat("conc", v1, v2)
            v4 = fvertex("globmeanpool", v3, x -> ONNXmutable.globalmeanpool(x, y -> dropdims(y, dims=(1,2))))
            v5 = dense("output", v4, 2, relu)

            test_named_graph(CompGraph(v0, v5), (2,3))
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
