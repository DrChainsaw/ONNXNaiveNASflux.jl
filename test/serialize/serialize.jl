
@testset "Structure" begin
    import ONNXNaiveNASflux: OnnxNode, input, output, optype, array

    function serdeser(p::T, convfun = cfun(p)) where T
        iob = PipeBuffer()
        ONNX.writeproto(iob, p)
        return convfun(ONNX.readproto(iob, T))
    end

    function serdeser(p::ONNX.ModelProto)
        iob = PipeBuffer();
        save(iob, p)
        return ONNXNaiveNASflux.extract(iob)
    end

    cfun(::ONNX.NodeProto) = np -> OnnxNode(np, ONNX.TensorProto[])
    cfun(::ONNX.TensorProto) = array
    cfun(::ONNX.GraphProto) = identity

    include("onnxruntime.jl")

    @testset "Nodes" begin
        using Statistics
        import ONNXNaiveNASflux: optype, actfuns, fluxlayers, invariantops
        using ONNXNaiveNASflux.NaiveNASflux
        import ONNXNaiveNASflux.NaiveNASflux: weights, bias
        import ONNXNaiveNASflux: AbstractProbe, nextname, newfrom, add!, genname, shape, nextshape, nodeprotos
        import Flux: unsqueeze
        struct NodeProbe{F, S} <: AbstractProbe
            name::String
            namefun::F
            shape::S
            protos::Vector{Any}
        end
        NodeProbe(name, namefun, shape=missing) = NodeProbe(name, namefun, shape, [])
        ONNXNaiveNASflux.add!(p::NodeProbe, n) = push!(p.protos, n)
        ONNXNaiveNASflux.nextname(p::NodeProbe) = p.namefun
        ONNXNaiveNASflux.newfrom(p::NodeProbe, pname, Δshape=identity) = NodeProbe(pname, p.namefun, nextshape(p, Δshape), p.protos)
        ONNXNaiveNASflux.newnamestrat(p::NodeProbe, f, pname = name(p)) = NodeProbe(pname, f, p.shape, p.protos)
        ONNXNaiveNASflux.name(p::NodeProbe) = p.name
        ONNXNaiveNASflux.shape(p::NodeProbe) = p.shape
        
        protos(p::ONNXNaiveNASflux.WrappedProbe) = protos(ONNXNaiveNASflux.unwrap(p))
        protos(p::NodeProbe) = p.protos

        @testset "Paramfree op $(tc.op) attrs: $(pairs(tc.attr))" for tc in (
            (op=:Relu, attr = Dict(), fd=actfuns),
            (op=:LeakyRelu, attr = Dict(), fd=actfuns),
            (op=:LeakyRelu, attr = Dict(:alpha => 0.05f0), fd=actfuns),
            (op=:Elu, attr = Dict(), fd=actfuns),
            (op=:Elu, attr = Dict(:alpha => 0.5f0), fd=actfuns),
            (op=:Selu, attr = Dict(), fd=actfuns),
            (op=:Selu, attr = Dict(:alpha => 1.5f0), fd=actfuns),
            (op=:Sigmoid, attr = Dict(), fd=actfuns),
            (op=:Tanh, attr = Dict(), fd=actfuns),
            (op=:GlobalAveragePool, attr=Dict(), fd=invariantops),
            (op=:MaxPool, attr=Dict(:kernel_shape=>(1,2), :pads=>(2,1), :strides=>(2,2)), fd=fluxlayers),
            (op=:AveragePool, attr=Dict(:kernel_shape=>(3,2), :pads=>(1,0), :strides=>(2,2)), fd=fluxlayers),
            (op=:Dropout, attr=Dict(:ratio => 0.2f0), fd=fluxlayers),
            )

            inprobe = NodeProbe("input", f -> "output")

            outprobe = tc.fd[tc.op](tc.attr)(inprobe)

            @test length(outprobe.protos) == 1

            res = serdeser(outprobe.protos[])

            @test input(res) == [name(inprobe)]
            @test output(res) == [name(outprobe)]
            @test optype(res) == tc.op
            @test name(res) == name(outprobe)

            mexprev(v, x) = x
            mexprev(v, x::Tuple) = reverse(x)
            mexprev(::Val{:pads}, x::Tuple) = ONNXNaiveNASflux.padexpand(Val(length(x)), x)
            for (k,v) in tc.attr
                for (exp, act) in zip(mexprev(Val(k), v), res.attribute[k])
                    @test exp == act
                end
            end
        end

        @testset "Dims method $(tc.ot)" for tc in (
            (f=cat, dims=1, expdims=1, ndims=2, ot=:Concat, axname=:axis),
            (f=mean, dims=(2, 3), expdims=[2, 3], ndims=4, ot=:ReduceMean, axname=:axes),
            (f=dropdims, dims=(3,), expdims=[3], ndims=3, ot=:Squeeze, axname=:axes),
            (f=unsqueeze, dims=3, expdims=[3], ndims=3, ot=:Unsqueeze, axname=:axes),
            )
            inprobe = NodeProbe("input", f -> "output", Tuple(1:tc.ndims))

            outprobe = tc.f(inprobe, dims=tc.dims)

            @test length(outprobe.protos) == 1

            res = serdeser(outprobe.protos[])

            @test input(res) == [name(inprobe)]
            @test output(res) == [name(outprobe)]
            @test optype(res) == tc.ot
            @test name(res) == name(outprobe)
            @test ONNXNaiveNASflux.numpy2fluxdim.(res.attribute[tc.axname], tc.ndims) == tc.expdims
            
            x = ones(Float32, ntuple(Returns(1), tc.ndims))
            invertex = convinputvertex(name(inprobe), 1, tc.ndims-1)
            @test ONNXNaiveNASflux.verts[tc.ot](name(res), [invertex], res.attribute)(x) == tc.f(x; dims=tc.dims)

            ortout, = onnxruntime_infer(x -> tc.f(x; dims=tc.dims), x)
            @test ortout == tc.f(x; dims=tc.dims)
        end

        @testset "Reshape" begin
            inprobe = NodeProbe("input", f -> "output", (:A, missing, 12))

            outprobe = reshape(inprobe, (0, 3, 2, Colon()))
            shapeout = shape(outprobe)

            @test length(outprobe.protos) == 2

            res = serdeser(outprobe.protos[1])
            newshape = serdeser(outprobe.protos[2])

            @test newshape == [-1, 2, 3, 0]

            @test input(res) == [name(inprobe), outprobe.protos[2].name]
            @test output(res) == [name(outprobe)]
            @test optype(res) == :Reshape
            @test name(res) == name(outprobe)
            @test ismissing(shapeout[end])
            @test collect(skipmissing(shapeout)) == [:A, 3, 2]
        end

        @testset "Flatten" begin
            import ONNXNaiveNASflux.NaiveNASflux.Flux: flatten
            inprobe = NodeProbe("input", f -> "output", (2, 3, 5))

            outprobe = Flux.flatten(inprobe)
            shapeout = shape(outprobe)

            @test length(outprobe.protos) == 1

            res = serdeser(outprobe.protos[1])

            @test input(res) == [name(inprobe)]
            @test output(res) == [name(outprobe)]
            @test optype(res) == :Flatten
            @test name(res) == name(outprobe)
            @test res.attribute[:axis] == -2

            op = ONNXNaiveNASflux.pseudotransparentops[optype(res)](res.attribute)
            indata = reshape(collect(1:2*3*5), 2,3,5)
            @test op(indata) == flatten(indata)
        end

        @testset "Pad expand" begin
            import ONNXNaiveNASflux: padexpand
            @test padexpand(Val(1), (1,)) == [1,1]
            @test padexpand(Val(2), (1,2)) == [2,1,2,1]
            @test padexpand(Val(3), (1,2,3)) == [3,2,1,3,2,1]

            @test padexpand(Val(1), (1,2)) == [1,2]
            @test padexpand(Val(2), (1,2,3,4)) == [3,1,4,2]
            @test padexpand(Val(3), (1,2,3,4,5,6)) == [5,3,1,6,4,2]
        end

        @testset "Layer with activation function $actfun" for actfun in (
            relu,
            leakyrelu,
            elu,
            selu,
            tanh,
        )
            @testset "$(tc.layer) node" for tc in (
                (layer=Dense(3,4, actfun), indata=reshape(collect(Float32, 1:12), :, 4) .- 3),
                (layer=Conv((1,2), 3=>4, actfun; pad=(2,1), stride=(1,2), dilation=3), indata=reshape(collect(Float32, 1:2*3*9*9), 9,9,3,2) .- 5),
                (layer=Conv((2,3), 3=>4, actfun; pad=(1,2,3,4), stride=(1,2), dilation=3), indata=reshape(collect(Float32, 1:2*3*9*9), 9,9,3,2) .- 10),
                (layer=ConvTranspose((3,3), 3=>4, actfun), indata=reshape(collect(Float32, 1:2*3*9*9), 9,9,3,2) .- 10),
                (layer=ConvTranspose((2,3), 3=>4, actfun; pad=(1,2,3,4), stride=(1,2), dilation=3), indata=reshape(collect(Float32, 1:2*3*9*9), 9,9,3,2) .- 10),
                )

                inprobe = NodeProbe("input", genname, shape(layertype(tc.layer), nin(tc.layer)))

                outprobe = tc.layer(inprobe)

                @test length(outprobe.protos) == 4

                wp,bp,lp,ap = Tuple(outprobe.protos)

                ln = serdeser(lp)
                an = serdeser(ap)
                w = serdeser(wp)
                b = serdeser(bp)

                @test size(w) == size(weights(tc.layer))
                @test size(b) == size(bias(tc.layer))

                @test w ≈ ONNXNaiveNASflux.flipweights(layertype(tc.layer), weights(tc.layer))
                @test b ≈ bias(tc.layer)

                ln.attribute[:activation] = actfuns[Symbol(optype(an))](an.attribute)
                res = fluxlayers[optype(ln)](ln.attribute, w, b)

                resout = res(tc.indata)
                expout = tc.layer(tc.indata)

                @test size(resout) == size(expout)
                @test resout ≈ expout

                ortout, = onnxruntime_infer(tc.layer, tc.indata)
                @test size(ortout) == size(expout)
                @test ortout ≈ expout
            end
        end

        @testset "$(tc.layer) node no bias no act" for tc in (
            (layer=Dense(randn(Float32, 2,3), false), indata=reshape(collect(Float32, 1:12), :, 4) .- 3),
            (layer=Conv((1,1), 2=>3; bias=false), indata=reshape(collect(Float32, 1:2*3), 1,1,2,3) .- 3),
            )

            inprobe = NodeProbe("input", genname, shape(layertype(tc.layer), nin(tc.layer)))

            outprobe = tc.layer(inprobe)

            @test length(outprobe.protos) == 2

            wp,lp= Tuple(outprobe.protos)

            ln = serdeser(lp)
            w = serdeser(wp)

            @test size(w) == size(weights(tc.layer))
            @test w ≈ ONNXNaiveNASflux.flipweights(layertype(tc.layer), weights(tc.layer))

            res = fluxlayers[optype(ln)](ln.attribute, w)

            resout = res(tc.indata)
            expout = tc.layer(tc.indata)

            @test size(resout) == size(expout)
            @test resout ≈ expout

            ortout, = onnxruntime_infer(tc.layer, tc.indata)
            @test size(ortout) == size(expout)
            @test ortout ≈ expout
        end

        @testset "$(tc.layer) node" for tc in (
            (layer=RNN(3 => 5, x -> Flux.elu(x, 0.1f0)), indata = reshape(collect(Float32, 1:24), :, 2, 4) .- 3),
            (layer=LSTM(4 => 3), indata = reshape(collect(Float32, 1:24), 4, 2, :) .- 3),
            )
            import ONNXNaiveNASflux.NaiveNASflux: hiddenweights

            inprobe = NodeProbe("input", genname, shape(layertype(tc.layer), nin(tc.layer)))
       
            # Since Flux LSTM outputs both hidden and cell state and onnxruntime seems to just output the hidden
            resultmap = get(tc, :resultmap, identity) 

            outprobe = resultmap(tc.layer(inprobe))

            @test length(protos(outprobe)) == 5

            wip,whp,bp,lp = Tuple(protos(outprobe))

            ln = serdeser(lp)
            wi = serdeser(wip)
            wh = serdeser(whp)
            b = serdeser(bp)

            res = fluxlayers[optype(ln)](ln.attribute, wi, wh, b)

            lt = layertype(tc.layer)
            @test size(weights(res)) == size(weights(tc.layer))
            @test size(hiddenweights(res)) == size(hiddenweights(tc.layer))
            @test size(bias(res)) == size(bias(res))

            @test weights(res) ≈ weights(tc.layer)
            @test hiddenweights(res) ≈ hiddenweights(tc.layer)
            @test bias(res) ≈ bias(res)

            resout = resultmap(res(tc.indata))
            expout = resultmap(tc.layer(tc.indata))

            @test size(resout) == size(expout)
            @test resout ≈ expout

            # Permute dims here since onnx wants [timesteps, batchsize, features] while flux wants [features, timesteps, batchsize]
            # Note that onnxruntime_infer reveres the dimensions for all inputs, so when we feed it [features, batchsize, timesteps]
            # it will do the reversal of the dimensions to [timesteps, batchsize, features]
            ortout, = onnxruntime_infer(resultmap ∘ tc.layer, permutedims(tc.indata, (1, 3, 2)))
            ortout = permutedims(ortout, (1, 3, 2))

            @test size(ortout) == size(expout)
            @test ortout ≈ expout
        end

        @testset "$(tc.layer) node" for tc in (
            (layer=BatchNorm(3, relu; initβ = i -> collect(Float32, 1:i), initγ = i -> collect(Float32, i:-1:1), eps=1e-3, momentum = 0.78), indata=reshape(collect(Float32, 1:2*3*3), 2,3,3,1) .- 10),
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

            resout = res(tc.indata)
            expout = tc.layer(tc.indata)

            @test size(resout) == size(expout)
            @test resout ≈ expout

            ortout, = onnxruntime_infer(tc.layer, tc.indata)
            @test size(ortout) == size(expout)
            @test ortout ≈ expout
        end

        @testset "$(tc.layer) node" for tc in (
            (layer=InstanceNorm(3, relu, affine=true), indata=reshape(collect(Float32, 1:2*3*3), 2,3,3,1) .- 10),
            (layer=InstanceNorm(3, relu, initβ = i -> collect(Float32, 1:i), initγ = i -> collect(Float32, i:-1:1), affine=true, track_stats=false, eps=1f-3), indata=reshape(collect(Float32, 1:2*3*3), 2,3,3,1) .- 10),
            )

            inprobe = NodeProbe("input", genname)
            outprobe = tc.layer(inprobe)
            @test length(outprobe.protos) == 4

            ln, γ, β, an = Tuple(serdeser.(outprobe.protos))

            @test size(β) == size(tc.layer.β)
            @test size(γ) == size(tc.layer.γ)

            @test β ≈ tc.layer.β
            @test γ ≈ tc.layer.γ

            ln.attribute[:activation] = actfuns[Symbol(optype(an))](an.attribute)
            res = fluxlayers[optype(ln)](ln.attribute, γ, β)

            @test string(res) == string(tc.layer)

            resout = res(tc.indata)
            expout = tc.layer(tc.indata)

            @test size(resout) == size(expout)
            @test resout ≈ expout

            ortout, = onnxruntime_infer(tc.layer, tc.indata)
            @test size(ortout) == size(expout)
            @test ortout ≈ expout
        end
    end

    @testset "Graphs" begin
        using ONNXNaiveNASflux.NaiveNASflux
        using ONNXNaiveNASflux: graphproto, modelproto, validate, AbstractVertex
        using NaiveNASflux: AbstractMutableComp

        dense(name, inpt::AbstractVertex, outsize, actfun=identity; kwargs...) = fluxvertex(name, Dense(nout(inpt), outsize, actfun), inpt; kwargs...)
        dense(inpt::AbstractVertex, outsize, actfun=identity; kwargs...) = fluxvertex(Dense(nout(inpt), outsize, actfun), inpt; kwargs...)

        convvertex(name, inpt::AbstractVertex, outsize, actfun=identity) = fluxvertex(name, Conv((1,1), nout(inpt) => outsize, actfun), inpt)

        bnvertex(name, inpt::AbstractVertex, actfun=identity) = fluxvertex(name, BatchNorm(nout(inpt), actfun), inpt)

        maxpvertex(name, inpt::AbstractVertex) = fluxvertex(name, MaxPool((2,2); pad=(1,0), stride=(1,2)), inpt)

        # TODO: Make which OP types shall be merged into a single vertex configurable... 
        gmpvertex(name, inpt::AbstractVertex) = invariantvertex(name, x -> dropdims(GlobalMeanPool()(x); dims=(1,2)), inpt)

        fvertex(name, inpt::AbstractVertex, f) = invariantvertex(name, f, inpt)

        recurrent_swap(x::AbstractArray{T, 3}) where T = permutedims(x, (1, 3, 2))
        recurrent_swap(x) = x

        test_outputs(res::Tuple, exp, args...) = test_outputs(res[1], exp, args...)
        test_outputs(res::Tuple, exp::Tuple, args...) = foreach((r, e) -> test_outputs(r, e, args...), res, exp)
        function test_outputs(res, exp, resmap=identity)
            resadj = resmap(res)
            @test size(resadj) == size(exp)
            @test resadj ≈ exp
        end

        function test_named_graph(g_org, extradims = (); timesteps=(), serialize_insizes=false, ortoutputmap=identity, ortinputmap=identity)
            outsize = nout(inputs(g_org)[])
            bs = 4
            indata = reshape(collect(Float32, 1:outsize*bs*prod(extradims)*prod(timesteps)), extradims..., outsize, timesteps..., :)

            gp_org = if serialize_insizes
                graphproto(g_org, name(inputs(g_org)[]) => size(indata); namestrat=ONNXNaiveNASflux.default_namestrat(g_org), name="testmodel")
            else
                graphproto(g_org; name="testmodel")
            end
            
            validate(modelproto(;graph=gp_org))
            gt_new = serdeser(gp_org)

            g_new = CompGraph(gt_new)

            @test name.(vertices(g_org)) == name.(vertices(g_new))
            @test nout.(vertices(g_org)) == nout.(vertices(g_new))
            @test nin.(vertices(g_org)) == nin.(vertices(g_new))

            expout = g_org(indata)
            resout = g_new(indata)

            test_outputs(resout, expout)

            ortout = onnxruntime_infer(g_org, ortinputmap(indata))
            test_outputs(ortout, expout, ortoutputmap)

            return g_new
        end

        @testset "Generic function" begin
            l1 = Dense(2, 3, elu)
            l2 = Dense(3, 2)
            function f(x, y)
                x1 = l1(x)
                x2 = l2(x1)
                return x2 .+ y
            end

            gp_sizes = graphproto(f, "x" => (2, missing), "y" => (2,))
            g_sizes = CompGraph(serdeser(gp_sizes))

            x = reshape(collect(Float32, 1:2*4), 2,4)
            y = Float32[5, 6]

            @test name.(vertices(g_sizes)) == ["x", "dense_0", "dense_1", "y", "add_0"]
            @test nout.(vertices(g_sizes)) == [2, 3, 2, 2, 2]
            @test nin.(vertices(g_sizes)) == [[], [2], [3], [], [2, 2]]

            @test g_sizes(x, y) ≈ f(x,y)

            function f(x)
                x1 = l1(x)
                return l2(x1)
            end

            gp_nosizes =  graphproto(f, "x" => missing)
            g_nosizes = CompGraph(serdeser(gp_nosizes))

            @test name.(vertices(g_nosizes)) == ["x", "dense_0", "dense_1"]
            @test g_nosizes(x) ≈ f(x)
        end

        @testset "Linear Dense graph" begin
            v0 = denseinputvertex("input", 3)
            v1 = dense("dense1", v0, 4, relu)
            v2 = dense("dense2", v1, 5, elu)
            v3 = dense("output", v2, 2)

            test_named_graph(CompGraph(v0, v3))
        end

        @testset "Linear Dense graph without names" begin
            v0 = denseinputvertex("input", 3)
            v1 = dense(v0, 4, selu)
            v2 = dense(v1, 5, relu)
            v3 = dense(v2, 2)

            g_org = CompGraph(v0, v3)

            gp_org = graphproto(g_org)
            gt_new = serdeser(gp_org)

            g_new = CompGraph(gt_new)

            @test name.(vertices(g_new)) == ["input_0", "dense_0", "dense_1", "dense_2"]

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "Linear Dense graph non-unique names" begin
            v0 = denseinputvertex("input", 3)
            v1 = dense("vv", v0, 4, selu)
            v2 = dense("vv", v1, 5, relu)
            g_org = CompGraph(v0, v2)

            gp_org = graphproto(g_org)
            gt_new = serdeser(gp_org)

            g_new = CompGraph(gt_new)

            @test name.(vertices(g_new)) == ["input_0", "dense_0", "dense_1",]

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "Linear Conv graph" begin
            v0 = conv2dinputvertex("input", 3)
            v1 = convvertex("conv1", v0, 4, selu)
            v2 = convvertex("conv2", v1, 5, elu)
            v3 = convvertex("output", v2, 2)

            test_named_graph(CompGraph(v0, v3), (2,3))
        end

        @testset "Linear Conv graph with global pooling" begin
            v0 = conv2dinputvertex("input", 3)
            v1 = convvertex("conv1", v0, 4, relu)
            v2 = convvertex("conv2", v1, 5, elu)
            v3 = gmpvertex("globalmeanpool", v2)
            v4 = dense("output", v3, 2)

            test_named_graph(CompGraph(v0, v4), (2,3))
        end

        @testset "Linear Conv graph with global pooling without names" begin
            v0 = conv2dinputvertex("input", 3)
            v1 = convvertex("", v0, 4, relu)
            v2 = gmpvertex("", v1)

            g_org = CompGraph(v0, v2)

            gp_org = graphproto(g_org)
            @test length(size(gp_org.output[])) == 2

            gt_new = serdeser(gp_org)

            g_new = CompGraph(gt_new)
            @test name.(vertices(g_new)) == ["input_0", "conv_0", "globalaveragepool_0"]

            indata = reshape(collect(Float32, 1:3*2*2*2), 2,2,3,2)
            @test size(g_org(indata)) == size(g_new(indata))
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "Linear Batchnorm and Conv graph with global pooling" begin
            v0 = conv2dinputvertex("input", 3)
            v1 = convvertex("conv", v0, 4, relu)
            v2 = bnvertex("batchnorm", v1, elu)
            v3 = gmpvertex("globalmeanpool", v2)
            v4 = dense("output", v3, 2, selu)

            test_named_graph(CompGraph(v0, v4), (4,6))
        end

        @testset "Linear Conv and MaxPool graph with global pooling" begin
            v0 = conv2dinputvertex("input", 3)
            v1 = maxpvertex("maxpool", v0)
            v2 = convvertex("conv", v1, 4, relu)
            v3 = gmpvertex("globalmeanpool", v2)
            v4 = dense("output", v3, 2, selu)

            test_named_graph(CompGraph(v0, v4), (2,3))
        end

        @testset "Dense graph with add" begin
            v0 = denseinputvertex("input", 3)
            v1 = dense("dense1", v0, 4, relu)
            v2 = dense("dense2", v0, 4)
            v3= "add" >> v1 + v2
            v4 = dense("output", v3, 2, selu)

            test_named_graph(CompGraph(v0, v4))
        end

        @testset "Dense graph with add without names" begin
            v0 = denseinputvertex("input", 3)
            v1 = dense(v0, 4, relu)
            v2 = dense(v0, 4)
            v3 = v1 + v2

            g_org = CompGraph(v0, v3)

            gp_org = graphproto(g_org)
            gt_new = serdeser(gp_org)

            g_new = CompGraph(gt_new)

            @test name.(vertices(g_new)) == ["input_0", "dense_0", "dense_1", "add_0"]

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "Dense graph with add and layerfun" begin
            import ONNXNaiveNASflux: create_vertex_default
            v0 = denseinputvertex("input", 3)
            v1 = dense(v0, 4, relu)
            v2 = dense(v0, 4)
            v3 = v1 + v2

            g_org = CompGraph(v0, v3)

            gp_org = graphproto(g_org)
            gt_new = serdeser(gp_org)

            callcnt = 0
            struct CntSpy <: AbstractMutableComp
                f
            end
            function (c::CntSpy)(x...)
                callcnt += 1
                return c.f(x...)
            end
            NaiveNASflux.wrapped(c::CntSpy) = c.f
            NaiveNASflux.layer(c::CntSpy) = layer(c.f)
            NaiveNASflux.layertype(c::CntSpy) = layertype(c.f)

            g_new = CompGraph(gt_new; vfun = (args...) -> create_vertex_default(args...;layerfun=CntSpy))

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            outdata = ones(Float32, nout(v3), size(indata, 2))
            Flux.train!((g,x,y) -> Flux.mse(g(x), y), g_new, [(indata, outdata)], Flux.Descent(0.6))
            @test callcnt == nvertices(g_new) - 1
        end

        @testset "Dense graph with cat" begin
            v0 = denseinputvertex("input", 3)
            v1 = dense("dense1", v0, 4, elu)
            v2 = dense("dense2", v0, 4)
            v3 = concat("conc", v1, v2)
            v4 = dense("output", v3, 2, relu)

            test_named_graph(CompGraph(v0, v4))
        end

        @testset "Dense graph with cat and layerfun" begin
            v0 = denseinputvertex("input", 3)
            v1 = dense("dense1", v0, 4, elu)
            v2 = dense("dense2", v0, 4)
            v3 = concat("conc", v1, v2)
            v4 = dense("output", v3, 2, relu)

            g_org = CompGraph(v0, v4)

            gp_org = graphproto(g_org)
            gt_new = serdeser(gp_org)

            callcnt = 0
            struct CntSpy <: AbstractMutableComp
                f
            end
            function (c::CntSpy)(x...)
                callcnt += 1
                return c.f(x...)
            end
            NaiveNASflux.wrapped(c::CntSpy) = c.f
            NaiveNASflux.layer(c::CntSpy) = layer(c.f)
            NaiveNASflux.layertype(c::CntSpy) = layertype(c.f)

            g_new = CompGraph(gt_new; vfun = (args...) -> create_vertex_default(args...;layerfun=CntSpy))

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            outdata = ones(Float32, nout(v4), size(indata, 2))

            Flux.train!((g,x,y) -> Flux.mse(g(x), y), g_new, [(indata, outdata)], Flux.Descent(0.6))
            @test callcnt == nvertices(g_new) - 1
        end

        @testset "Dense graph with constant scalar op $op" for op in (+, *)
            v0 = denseinputvertex("input", 3)
            v1 = dense("dense1", v0, 4, elu)
            v2 = fvertex("scale", v1, x -> op.(0.5f0, x))
            v3 = dense("output", v2, 2, relu)

            test_named_graph(CompGraph(v0, v3))
        end

        @testset "Dense graph with constant elemwise array op $op" for op in (+, *)
            v0 = denseinputvertex("input", 3)
            v1 = dense("dense1", v0, 2, elu)
            v2 = fvertex("scale", v1, x -> op.(Float32[0.5, 0.1], x))
            v3 = dense("output", v2, 3, relu)

            test_named_graph(CompGraph(v0, v3))
        end

        @testset "Dense graph with free constant" begin
            import ONNXNaiveNASflux: sourcevertex_with_outputs
            v0 = denseinputvertex("input", 3)
            v1 = dense("dense", v0, 2, elu)
            # Constant must be array for generic check to pass as ONNX opset only supports constant tensors
            # Name is not preserved when serializing SourceVertex. We just sneakily set it to what the autoselected name will be
            v2 =  sourcevertex_with_outputs([213], "constant")

            test_named_graph(CompGraph([v0], [v1,v2]))
        end

        @testset "Conv and batchnorm graph with cat" begin
            v0 = conv2dinputvertex("input", 3)
            v1 = convvertex("conv", v0, 2, elu)
            v2 = bnvertex("batchnorm", v0)
            v3 = concat("conc", v1, v2)
            v4 = gmpvertex("globalmeanpool", v3)
            v5 = dense("output", v4, 2, relu)

            test_named_graph(CompGraph(v0, v5), (2,3))
        end

        @testset "Dense graph with cat without names" begin
            v0 = denseinputvertex("input", 3)
            v1 = dense(v0, 4, relu)
            v2 = dense(v0, 4)
            v3 = concat("conc", v1, v2)

            g_org = CompGraph(v0, v3)

            gp_org = graphproto(g_org)
            gt_new = serdeser(gp_org)

            g_new = CompGraph(gt_new)

            @test name.(vertices(g_new)) == ["input_0", "dense_0", "dense_1", "concat_0"]

            indata = reshape(collect(Float32, 1:3*4), nout(v0), :)
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "RNN to LSTM" begin
            v0 = rnninputvertex("input", 3)
            v1 = fluxvertex("rnn", RNN(nout(v0) => 4), v0)
            v2 = fluxvertex("lstm", LSTM(nout(v1) => 5), v1)

            test_named_graph(CompGraph(v0, v2); timesteps=2, ortoutputmap=recurrent_swap, ortinputmap=recurrent_swap)
        end

        @testset "Recurrent to Dense" begin
            v0 = rnninputvertex("input", 3)
            v1 = fluxvertex("rnn", RNN(nout(v0) => 4), v0)
            v2 = fluxvertex("lstm", LSTM(nout(v1) => 5), v1)
            v3 = dense("dense", v2, 2, elu)

            g_org = CompGraph(v0, v3)
            g_new = CompGraph(serdeser(graphproto(g_org)))

            @test name.(vertices(g_new)) == name.(vertices(g_org))

            indata = reshape(collect(Float32, 1:3*5*7), 3,5,7)

            expout = g_org(indata)
            resout = g_new(indata)

            @test size.(expout) == size.(resout)
            @test expout ≈ resout

            ortout, = onnxruntime_infer(g_org, recurrent_swap(indata))
            expout_s = reshape(recurrent_swap(expout), nout(v3), :)

            @test size(expout_s) == size(ortout)
            @test expout_s ≈ ortout
        end

        @testset "RNN select multiple outputs" begin
            import ONNXNaiveNASflux: OutputSelection, _rnn_output_selection, AddSingletonDim

            # [seq_length, num_directions, batch_size, hidden_size] gets reversed to
            # [hidden_size, batch_size, num_directions, seq_length]
            # Need to swap batch_size and seq_length to match Flux output with added dim 3
            # Maybe this is something ONNXNaiveNASflux needs to adjust for in order to support imported models...
            recurrent_swap_onnx(x::AbstractArray{T, 4}) where T = permutedims(x, (1, 4, 3, 2))
            recurrent_swap_onnx(x) = x

            @testset "Select first and second" begin
                v0 = rnninputvertex("input", 3)
                v1 = fluxvertex("rnn", RNN(nout(v0) => 4), v0; layerfun=l -> OutputSelection(ntuple(_rnn_output_selection, 2), l))

                test_named_graph(CompGraph(v0, v1); timesteps=2, ortoutputmap=recurrent_swap, ortinputmap=recurrent_swap)
            end

            @testset "Select first and second with direction dim" begin
                v0 = rnninputvertex("input", 3)
                v1 = fluxvertex("rnn", RNN(nout(v0) => 3), v0; layerfun=l -> AddSingletonDim(3, OutputSelection(ntuple(_rnn_output_selection, 2), l)))
                
                test_named_graph(CompGraph(v0, v1); timesteps=2, ortoutputmap=recurrent_swap_onnx, ortinputmap=recurrent_swap)
            end

        end

        @testset "LSTM select multiple outputs" begin
            import ONNXNaiveNASflux: OutputSelection, _lstm_output_selection, AddSingletonDim

            # [seq_length, num_directions, batch_size, hidden_size] gets reversed to
            # [hidden_size, batch_size, num_directions, seq_length]
            # Need to swap batch_size and seq_length to match Flux output with added dim 3
            recurrent_swap_onnx(x::AbstractArray{T, 4}) where T = permutedims(x, (1, 4, 3, 2))
            recurrent_swap_onnx(x) = x

            @testset "Select first and second" begin
                v0 = rnninputvertex("input", 3)
                v1 = fluxvertex("lstm", LSTM(nout(v0) => 4), v0; layerfun=l -> OutputSelection(ntuple(_lstm_output_selection, 2), l))

                test_named_graph(CompGraph(v0, v1); timesteps=2, ortoutputmap=recurrent_swap, ortinputmap=recurrent_swap)
            end

            @testset "Select first, second and third" begin
                v0 = rnninputvertex("input", 3)
                v1 = fluxvertex("lstm", LSTM(nout(v0) => 4), v0; layerfun=l -> OutputSelection(ntuple(_lstm_output_selection, 3), l))

                @test_throws ArgumentError CompGraph(serdeser(graphproto(CompGraph(v0, v1))))
                # Tested ok in Flux 0.15
                #test_named_graph(CompGraph(v0, v1); timesteps=2, ortoutputmap=recurrent_swap, ortinputmap=recurrent_swap)
            end

            @testset "Select first and second with direction dim" begin
                v0 = rnninputvertex("input", 3)
                v1 = fluxvertex("lstm", LSTM(nout(v0) => 3), v0; layerfun=l -> AddSingletonDim(3, OutputSelection(ntuple(_lstm_output_selection, 2), l)))
                
                test_named_graph(CompGraph(v0, v1); timesteps=2, ortoutputmap=recurrent_swap_onnx, ortinputmap=recurrent_swap)
            end

            @testset "Select first, second and third with direction dim" begin
                v0 = rnninputvertex("input", 3)
                v1 = fluxvertex("lstm", LSTM(nout(v0) => 4), v0; layerfun=l -> AddSingletonDim(3, OutputSelection(ntuple(_lstm_output_selection, 3), l)))

                @test_throws ArgumentError CompGraph(serdeser(graphproto(CompGraph(v0, v1))))
                # Tested ok in Flux 0.15
                # test_named_graph(CompGraph(v0, v1); timesteps=2, ortoutputmap=recurrent_swap_onnx, ortinputmap=recurrent_swap)
            end
        end

        @testset "Graph two inputs two outputs" begin
            vins = denseinputvertex.(["in1", "in2"], 3)
            v1 = "add" >> vins[1] + vins[2]
            v2 = concat("conc", vins[1], vins[2])

            g_org = CompGraph(vins, [v1, v2])
            g_new = CompGraph(serdeser(graphproto(g_org)))

            @test name.(vertices(g_org)) == name.(vertices(g_new))

            indata1 = reshape(collect(Float32, 1:3*4), nout(vins[1]), :)
            indata2 = indata1 .* -0.5
            @test g_org(indata1, indata2) == g_new(indata1, indata2)
        end

        @testset "Conv to reshape to Dense" begin
            v0 = conv2dinputvertex("input", 3)
            v1 = convvertex("conv", v0, 4, relu)
            v2 = absorbvertex("reshape", ONNXNaiveNASflux.Reshape((12, Colon())), v1; traitdecoration=SizePseudoTransparent)
            v3 = dense("dense", v2, 3, elu)

            test_named_graph(CompGraph(v0, v3), (2,3))
        end

        @testset "Conv to Flatten to Dense" begin
            using ONNXNaiveNASflux: MeasureNout
            v0 = conv2dinputvertex("input", 3)
            v1 = convvertex("conv", v0, 4, relu)
            v2 = absorbvertex("flatten",MeasureNout(ONNXNaiveNASflux.Flatten(3); outsize=24), v1; traitdecoration=SizePseudoTransparent)
            v3 = dense("dense", v2, 3, elu)

            test_named_graph(CompGraph(v0, v3), (2,3); serialize_insizes=true)
        end

        @testset "Conv to flatten to Dense" begin
            using ONNXNaiveNASflux: MeasureNout
            v0 = conv2dinputvertex("input", 3)
            v1 = convvertex("conv", v0, 4, relu)
            v2 = absorbvertex("flatten", MeasureNout(Flux.flatten; actdim=1, outsize=24), v1; traitdecoration=SizePseudoTransparent)
            v3 = dense("dense", v2, 3, elu)

            test_named_graph(CompGraph(v0, v3), (2,3); serialize_insizes=true)
        end

        @testset "Infer shapes" begin

            function remodel(g, args...=missing)
                pb = PipeBuffer()
                save(pb, g, args...)
                load(pb)
            end
            
            @testset "Batchnorm -> Conv graph" begin
                v0 = conv2dinputvertex("input", 3)    
                v1 = bnvertex("v1", v0)
                v2 = convvertex("v2", v1, 2)

                g = remodel(CompGraph(v0, v2))
                @test layertype(inputs(g)[1]) == layertype(v0)
            end

            @testset "Two conv graph" begin
                v0 = conv2dinputvertex("input", 3)    
                v1a = convvertex("v1a", v0, 4)
                v1b = convvertex("v1b", v0, 2)
                v2 = concat("v2", v1a, v1b)

                g_org = g = remodel(CompGraph(v0, v2))
                @test layertype(inputs(g)[1]) == layertype(v0)
            end

            @testset "Concat path" begin
                v0 = denseinputvertex("input", 3)
                v1 = dense("v1", v0, 4, elu)
                v2 = concat(v1, v0)    

                g = remodel(CompGraph(v0, v2))
                @test layertype(inputs(g)[1]) == layertype(v0)
            end

            @testset "Shortcut to globpool -> dense" begin
                v0 = conv2dinputvertex("input", 3)    
                v1 = convvertex("v1", v0, 2)
                v2 = concat("v2", v1, v0)
                v3 = gmpvertex("globalmeanpool", v2)
                v4 = dense("v4", v3, 4)

                g = remodel(CompGraph(v0, v4))
                @test layertype(inputs(g)[1]) == layertype(v0)
            end
        end
    end

    @testset "Chains" begin
        import ONNXNaiveNASflux: modelproto

        function remodel(m, args...; kwargs...)
            pb = PipeBuffer()
            save(pb, m, args...; kwargs...)
            return load(pb)
        end

        @testset "Simple Chain" begin
            org = Chain(Dense(1 => 2, relu), Dense(2 => 3, sigmoid), Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
        end

        @testset "Simple Named Chain" begin
            org = Chain(layer1 = Dense(1 => 2, relu), layer2 = Dense(2 => 3, sigmoid), layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "layer3"]
        end

        @testset "Simple Named Chain with name_runningnr" begin
            org = Chain(layer1 = Dense(1 => 2, relu), layer2 = Dense(2 => 3, sigmoid), layer3 = Dense(3 => 4))
            res = remodel(org; namestrat=ONNXNaiveNASflux.name_runningnr())

            x = randn(Float32, 1, 4)            
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org, namestrat=ONNXNaiveNASflux.name_runningnr())
            @test name.(mp.graph.node) == ["dense_0", "dense_0_relu", "dense_1", "dense_1_sigmoid", "dense_2"]
        end

        @testset "Nested Named Chain" begin
            org = Chain(
                        layer1 = Dense(1 => 2, relu), 
                        layer2 = Dense(2 => 3, sigmoid), 
                        inner = Chain(
                            Dense(3 => 3, tanh),
                            Dense(3=>3)), 
                        layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) ==  ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "inner[1]", "inner[1]_tanh", "inner[2]", "layer3"]
        end

        @testset "Nested Named Chain Array" begin
            org = Chain(
                        layer1 = Dense(1 => 2, relu), 
                        layer2 = Dense(2 => 3, sigmoid), 
                        inner = Chain([
                            Dense(3 => 3, tanh),
                            Dense(3=>3)]), 
                        layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) ==  ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "inner[1]", "inner[1]_tanh", "inner[2]", "layer3"]
        end

        @testset "Nested Named Chain Named Inner" begin
            org = Chain(
                        layer1 = Dense(1 => 2, relu), 
                        layer2 = Dense(2 => 3, sigmoid), 
                        inner = Chain(
                            ilayer1 = Dense(3 => 3, tanh),
                            ilayer2 = Dense(3=>3)), 
                        layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "inner.ilayer1", "inner.ilayer1_tanh", "inner.ilayer2", "layer3"]
        end

        @testset "Nested Chain Named Inner" begin
            org = Chain(
                        Dense(1 => 2, relu), 
                        Dense(2 => 3, sigmoid), 
                        Chain(
                            ilayer1 = Dense(3 => 3, tanh),
                            ilayer2 = Dense(3=>3)), 
                        Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["dense_0", "dense_0_relu", "dense_1", "dense_1_sigmoid", "dense_2", "dense_2_tanh", "dense_3", "dense_4"]
        end

        @testset "Chain Parallel" begin
            org = Chain(
                        Dense(1 => 2, relu), 
                        Dense(2 => 3, sigmoid), 
                        Parallel(+, 
                            Chain(
                                Dense(3 => 3, tanh),
                                Dense(3 => 3)),
                            Dense(3 => 3, elu),
                            ),
                        Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["dense_0", "dense_0_relu", "dense_1", "dense_1_sigmoid", "dense_2", "dense_2_tanh", "dense_3", "dense_4", "dense_4_elu", "add_0", "dense_5"]
        end

        @testset "Named Chain Parallel" begin
            org = Chain(
                        layer1 = Dense(1 => 2, relu), 
                        layer2 = Dense(2 => 3, sigmoid), 
                        fork = Parallel(+, 
                            Chain(
                                Dense(3 => 3, tanh),
                                Dense(3 => 3)),
                            Dense(3 => 3, elu),
                            ),
                        layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "fork[1][1]", "fork[1][1]_tanh", "fork[1][2]", "fork[2]", "fork[2]_elu", "fork.connection", "layer3"] 
        end

        @testset "Named Chain Named Parallel" begin
            org = Chain(
                        layer1 = Dense(1 => 2, relu), 
                        layer2 = Dense(2 => 3, sigmoid), 
                        fork = Parallel(+, 
                            path1 = Chain(
                                Dense(3 => 3, tanh),
                                Dense(3 => 3)),
                            path2 = Dense(3 => 3, elu),
                            ),
                        layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "fork.path1[1]", "fork.path1[1]_tanh", "fork.path1[2]", "fork.path2", "fork.path2_elu", "fork.connection", "layer3"] 
        end


        @testset "Named Chain Parallel Named Chain" begin
            org = Chain(
                        layer1 = Dense(1 => 2, relu), 
                        layer2 = Dense(2 => 3, sigmoid), 
                        fork = Parallel(+, 
                            Chain(
                                l1 = Dense(3 => 3, tanh),
                                l2 = Dense(3 => 3)
                                ),
                            Chain(
                                Dense(3 => 3, elu),
                                Dense(3 => 3, leakyrelu)
                                )
                            ),
                        layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "fork[1].l1", "fork[1].l1_tanh", "fork[1].l2", "fork[2][1]", "fork[2][1]_elu", "fork[2][2]", "fork[2][2]_leakyrelu", "fork.connection", "layer3"] 
        end

        @testset "Named Chain SkipConnection" begin
            org = Chain(
                        layer1 = Dense(1 => 2, relu), 
                        layer2 = Dense(2 => 3, sigmoid), 
                        fork = SkipConnection( 
                            Chain(
                                Dense(3 => 3, tanh),
                                Dense(3 => 3)),
                            +),
                        layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "fork.layers[1]", "fork.layers[1]_tanh", "fork.layers[2]", "fork.connection", "layer3"]
        end

        @testset "Named Chain CompGraph" begin
            org = Chain(
                        layer1 = Dense(1 => 2, relu), 
                        layer2 = Dense(2 => 3, sigmoid), 
                        graph = let 
                            iv = denseinputvertex("graphin", 3)
                            v1 = fluxvertex("v1", Dense(3 => 3, elu), iv)
                            v2 = "v2" >> iv + v1
                            CompGraph(iv, v2)                            
                        end,
                        layer3 = Dense(3 => 4))
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["layer1", "layer1_relu", "layer2", "layer2_sigmoid", "graph.v1", "graph.v1_elu", "graph.v2", "layer3"]
        end

        @testset "CompGraph Named Chain" begin
            org = let 
                iv = denseinputvertex("graphin", 1)
                v1 = fluxvertex("v1", Dense(1 => 3, elu), iv)
                v2 = invariantvertex("chain", Chain(l1 = Dense(3 => 3, relu), l2 = Dense(3 => 3, tanh)), v1)
                v3 = "v3" >> v1 + v2
                CompGraph(iv, v3)
            end
            res = remodel(org)

            x = randn(Float32, 1, 4)
            @test org(x) == res(x) ≈ only(onnxruntime_infer(org, x))
            mp = modelproto(org)
            @test name.(mp.graph.node) == ["v1", "v1_elu", "chain.l1", "chain.l1_relu", "chain.l2", "chain.l2_tanh", "v3"] 
        end
    end

    @testset "Models" begin
        import ONNXNaiveNASflux: modelproto, sizes, clean_size, OutputSelection

        @testset "Generic function infer" begin
            _f(x, y) = x .+ y
            f(x, y) = _f(x,y)
            f(x::Matrix{Int}, y::Matrix{Int}) = _f(x, y)

            mp = modelproto(f)
            mt = serdeser(mp)
            ss = sizes(mt)

            @test length(ss["data_0"]) == 2
            @test length(ss["data_1"]) == 2

            g = @test_logs (:warn, r"No valid input sizes provided.") load(mt)
            g([1,2], [3,4]) == f([1,2], [3,4])
        end

        @testset "CompGraph infer" begin
            vis = denseinputvertex.("in", [2, 2])
            g_org = CompGraph(vis, +(vis...))

            mp = modelproto(g_org)
            mt = serdeser(mp)
            ss = sizes(mt)

            @test length(ss["in_0"]) == 2
            @test length(ss["in_1"]) == 2

            g_new = load(mt)
            g_org([1,2], [3,4]) == g_new([1,2], [3,4])
        end

        @testset "$(cfun(l)) infer" for (l, expshape) in (
            (Dense(2 => 3), (2,0)),
            (Conv((1, 1), 2=>3), (0,0,2,0)), 
            (RNN(2 => 3), (2,0,0)), 
            (LSTM(2 => 3), (2,0,0)), 
            (SkipConnection(Dense(2 => 2), +), (2,0)),
            ), cfun in (Chain, l -> Chain(BatchNorm(nin(l)[]), l))

            c_org = cfun(l)

            mt = modelproto(c_org) |> serdeser
            ss = clean_size(sizes(mt))
 
            @test ss["data_0"] == expshape
        end
    end

    @testset "Save to file" begin
        using ONNXNaiveNASflux.NaiveNASflux
        function tryfile(filename, args...; kwargs...)
            try
                save(filename, args...; kwargs...)
                return load(filename)
            finally
                rm(filename;force=true)
            end
        end

        @testset "Generic function no pars" begin
            f = (x,y) -> x + y
            g = tryfile("generic_function_no_pars.onnx", f)
            @test name.(vertices(g)) == ["data_0", "data_1", "add_0"]
            @test g(1,3) == f(1, 3)
        end

        @testset "Generic function sizes" begin
            f = (x,y) -> x + y
            g = tryfile("generic_function_sizes.onnx", f, (2,missing), (2,missing))
            nout.(vertices(g)) == [2, 2]
            @test g([1,3], [2, 4]) == f([1,3], [2, 4])
        end

        @testset "Simple graph no pars" begin
            v0 = denseinputvertex("in", 3)
            v1 = fluxvertex("dense1", Dense(3, 2, relu), v0)
            v2 = fluxvertex("dense2", Dense(2, 3), v1)
            g_org = CompGraph(v0, v2)

            g_new = tryfile("simple_graph_no_pars.onnx", g_org)

            @test name.(vertices(g_org)) == name.(vertices(g_new))
            @test nout.(vertices(g_org)) == nout.(vertices(g_new))
            @test nin.(vertices(g_org)) == nin.(vertices(g_new))

            indata = reshape(collect(Float32, 1:3*2), 3,2)
            @test g_org(indata) ≈ g_new(indata)
        end

        @testset "Simple graph namestrat" begin
            v0 = conv2dinputvertex("in", 3)
            v1 = fluxvertex("conv", Conv((1,2), 3 => 4, relu), v0)
            v2 = fluxvertex("bn", BatchNorm(4, elu), v1)
            g_org = CompGraph(v0, v2)

            ng = ONNXNaiveNASflux.name_runningnr()
            ns(::NaiveNASlib.MutationVertex) = n -> ng
            ns(n) = ng(n)

            g_new = tryfile("simple_graph_namestrat.onnx", g_org; namestrat=ns)

            @test name.(vertices(g_new)) == ["in_0", "conv_0", "batchnorm_0"]
            @test nout.(vertices(g_org)) == nout.(vertices(g_new))
            @test nin.(vertices(g_org)) == nin.(vertices(g_new))

            indata = reshape(collect(Float32, 1:1*2*3*4), 1,2,3,4)
            @test g_org(indata) ≈ g_new(indata)
        end
    end

    @testset "Allowed input shapes" begin
        function remodel(m, args...; assertwarn=true)
            pb = PipeBuffer()
            save(pb, m, args...)
            if assertwarn
                return @test_logs (:warn, r"No valid input sizes") load(pb)
            end
            return load(pb)
        end

        @testset "Allowed input shapes op: $(tc[1])" for tc in (
            (Dense(2 => 3), ((2, 3), (2, missing), missing, (missing, missing)), (2, 2)),
            (Conv((1,), 2=>3), ((1,2,1), (1, missing, missing), missing, ntuple(i -> missing, 3)), (1,2,1)),
            (Conv((1,1), 2=>3), ((1,1,2,1), (1, missing, missing, missing), missing, ntuple(i -> missing, 4)), (1,1,2,1)),
            (RNN(2 => 3), ((2,3,1), missing, ntuple(i -> missing, 3), ntuple(i -> missing ,4)), (2, 3))
        )     
            op, testsizes, validsize = tc
            inpt = ones(Float32, validsize)

            g1 = remodel(op; assertwarn=false)
            @test g1(inpt) == op(inpt)
            @testset "Inputshape $s" for s in testsizes
                assertwarn = s isa Tuple && length(s) != length(shape(layertype(op), 1))
                g = remodel(op, s; assertwarn)
                @test g(inpt) == op(inpt)
            end
        end

        @testset "Allowed input shapes op: +" begin
            in1, in2 = ones(3), ones(3)
            op = (x,y) -> x .+ y
            g1 = remodel(op; assertwarn=false)
            @test g1(in1, in2) == in1 .+ in2
            @testset "Inputshape $s1, $s2" for (s1, s2, assertwarn) in (
                ((3,), (3,), false),
                ((missing,), (missing,), true),
                (missing, missing, false),
                )
                g = remodel(op, s1, s2; assertwarn)
                @test g(in1, in2) == in1 .+ in2
            end
        end
        
        @testset "Disallowed input shapes op: $(tc[1])" for tc in (
            (Dense(2 => 3), ((2,), (3, 1), (missing, ), (1,2,3,4))),
            (Conv((1,), 2 => 3), ((2,), (missing, ), (1,1,1), (2,3,4,5,6), (2,3,4,5))),
            (Conv((1,1), 2 => 3), ((2,), (missing, ), (1,1,1,1), (2,3,4,5,6), (2,3,4))),
            (MaxPool((2,2)), ((2,), (missing, ), (1,1,1), (2,3,4,5,6))),
            (RNN(2 => 3), ((2, ), (missing,), (2,1), (1,2,3), (2,3,4,5,6)))
        )
            op, testsizes = tc
            @testset "Inputshape $s" for s in testsizes
                @test_throws DimensionMismatch remodel(op, s)
            end
        end
    end
end
