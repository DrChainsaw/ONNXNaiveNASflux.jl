import ONNXNaiveNASflux: fluxlayers, sources, actfuns, invariantops, pseudotransparentops, optype, nodes, array, params, layerfuns
using ONNXNaiveNASflux.NaiveNASflux

# Logging to avoid CI timeouts
@info "  Test padding and sources"

@testset "Read padding" begin
    import ONNXNaiveNASflux: prev

    @test prev(2) == 2
    @test prev([1,2]) == [1,2]
    @test prev([1,2,3,4]) == [2,4,1,3]
    @test prev([1,2,3,4,5,6]) == [3,6,2,5,1,4]
end

@testset "Sources" for tc in
    (
    (name="test_constant", ninputs=0, noutputs=1),
    )

    model, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) op $(optype(node))" for node in nodes(gb)
        @test haskey(sources, optype(node))
        res = sources[optype(node)](node.attribute, params(node)...)

        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end

    @testset "$(tc.name) graph" begin
        cg = load(model)
        res = cg() 
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]

        # Also test that it we get the same thing by serializing and then deserializing
        io = PipeBuffer()
        save(io, cg)
        cg = load(io)
        res = cg() 
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end
end

@info "  Test Flux layers"

# ONNX puts batch dimension in the middle and time steps last while Flux has time steps in middle and batch last
recurrent_remap(x::AbstractArray{T, 3}) where T = permutedims(x, (1, 3, 2))
recurrent_remap(x) = x


@testset "Fluxlayer $(tc.name)" for tc in
    (
    (name="test_averagepool_1d_default", ninputs=1, noutputs=1),
    # (name="test_averagepool_2d_ceil", ninputs=1, noutputs=1), Not supported!
    (name="test_averagepool_2d_default", ninputs=1, noutputs=1),
    #(name="test_averagepool_2d_pads", ninputs=1, noutputs=1), Not supported!
    (name="test_averagepool_2d_strides", ninputs=1, noutputs=1),
    (name="test_averagepool_3d_default", ninputs=1, noutputs=1),
    (name="test_basic_conv_with_padding", ninputs=2, noutputs=1),
    (name="test_basic_conv_without_padding", ninputs=2, noutputs=1),
    (name="test_batchnorm_epsilon", ninputs=5, noutputs=1),
    (name="test_batchnorm_example", ninputs=5, noutputs=1),
    (name="test_conv_with_strides_and_asymmetric_padding", ninputs=2, noutputs=1),
    (name="test_conv_with_strides_no_padding", ninputs=2, noutputs=1),
    (name="test_conv_with_strides_padding", ninputs=2, noutputs=1),
    (name="test_convtranspose", ninputs=2, noutputs=1),
    (name="test_convtranspose_1d", ninputs=2, noutputs=1),
    (name="test_convtranspose_3d", ninputs=2, noutputs=1),
    (name="test_convtranspose_dilations", ninputs=2, noutputs=1),
    #(name="test_convtranspose_kernel_shape", ninputs=2, noutputs=1), Not supported!
    #(name="test_convtranspose_output_shape", ninputs=2, noutputs=1), Not supported!
    #(name="test_convtranspose_pad", ninputs=2, noutputs=1), Not supported!
    (name="test_convtranspose_pads", ninputs=2, noutputs=1),
    #(name="test_convtranspose_with_kernel", ninputs=2, noutputs=1), Not supported!
    (name="test_dropout_default", ninputs=1, noutputs=1),
    (name="test_dropout_random", ninputs=1, noutputs=1),
    #(name="test_gemm_all_attributes", ninputs=3, noutputs=1), Not supported!
    (name="test_gemm_alpha", ninputs=3, noutputs=1),
    (name="test_gemm_beta", ninputs=3, noutputs=1),
    #(name="test_gemm_default_matrix_bias", ninputs=3, noutputs=1), Not supported!
    (name="test_gemm_default_no_bias", ninputs=2, noutputs=1),
    (name="test_gemm_default_scalar_bias", ninputs=3, noutputs=1),
    (name="test_gemm_default_single_elem_vector_bias", ninputs=3, noutputs=1),
    (name="test_gemm_default_vector_bias", ninputs=3, noutputs=1),
    (name="test_gemm_default_zero_bias", ninputs=3, noutputs=1),
    #(name="test_gemm_transposeA", ninputs=3, noutputs=1), Not supported!
    (name="test_gemm_transposeB", ninputs=3, noutputs=1),
    (name="test_instancenorm_epsilon", ninputs=3, noutputs=1),
    (name="test_instancenorm_example", ninputs=3, noutputs=1),
    (name="test_lstm_defaults", ninputs=3, noutputs=1, inputmap=recurrent_remap, outputmap=recurrent_remap, resultmap=recurrent_remap),
    (name="test_lstm_with_initial_bias", ninputs=4, noutputs=1, inputmap=recurrent_remap, outputmap=recurrent_remap, resultmap=recurrent_remap),
    # (name="test_lstm_with_peepholes", ninputs=8, noutputs=1), Not supported!
    (name="test_maxpool_1d_default", ninputs=1, noutputs=1),
    #(name="test_maxpool_2d_ceil", ninputs=1, noutputs=1), Not supported!
    (name="test_maxpool_2d_default", ninputs=1, noutputs=1),
    #(name="test_maxpool_2d_dilations", ninputs=1, noutputs=1), Not supported!
    #(name="test_maxpool_2d_pads", ninputs=1, noutputs=1), Not supported!
    (name="test_maxpool_2d_strides", ninputs=1, noutputs=1),
    (name="test_maxpool_3d_default", ninputs=1, noutputs=1),
    (name="test_maxpool_3d_default", ninputs=1, noutputs=1),
    (name="test_rnn_seq_length", ninputs=4, noutputs=1, inputmap=recurrent_remap, outputmap=recurrent_remap, resultmap=recurrent_remap),
    )

    model, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    inputs = map(get(tc, :inputmap, identity), inputs)
    outputs = map(get(tc, :outputmap, identity), outputs)

    resultmap = get(tc, :resultmap, identity)

    @testset "$(tc.name) op $(optype(node))" for node in nodes(gb)
        @test haskey(fluxlayers, optype(node))
        op = fluxlayers[optype(node)](node.attribute, params(node)...)
        opwrapped = op |> get(layerfuns, optype(node), Returns(identity))(node.attribute, params(node)...)

        res = resultmap(opwrapped(Float32.(inputs[1])))
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end

    @testset "$(tc.name) graph" begin
        cg = load(model)
        res = resultmap(cg(Float32.(inputs[1])))
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]

        # Also test that it we get the same thing by serializing and then deserializing
        io = PipeBuffer()
        save(io, cg)
        cg = load(io)
        res = resultmap(cg(Float32.(inputs[1])))
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end
end

@info "  Test Flux activation functions"

@testset "Activation functions $(tc.name)" for tc in
    (
    (name="test_elu", ninputs=1, noutputs=1),
    (name="test_elu_default", ninputs=1, noutputs=1),
    (name="test_elu_example", ninputs=1, noutputs=1),
    (name="test_relu", ninputs=1, noutputs=1),
    (name="test_leakyrelu", ninputs=1, noutputs=1),
    (name="test_leakyrelu_default", ninputs=1, noutputs=1),
    (name="test_leakyrelu_example", ninputs=1, noutputs=1),
    (name="test_selu", ninputs=1, noutputs=1),
    (name="test_selu_default", ninputs=1, noutputs=1),
    (name="test_selu_example", ninputs=1, noutputs=1),
    (name="test_sigmoid", ninputs=1, noutputs=1),
    (name="test_sigmoid_example", ninputs=1, noutputs=1),
    )

    model, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) op $(optype(node))" for node in nodes(gb)
        @test haskey(actfuns, optype(node))
        op = actfuns[optype(node)](node.attribute, params(node)...)
        @test op.(inputs[1]) ≈ outputs[1]

        @test haskey(invariantops, optype(node))
        bcop = invariantops[optype(node)](node.attribute, params(node)...)
        @test bcop(inputs[1]) ≈ outputs[1]

    end
end

@info "  Test stateless ops"

@testset "Invariant op $(tc.name)" for tc in
    (
    (name="test_flatten_axis0", ninputs=1, noutputs=1, fd=pseudotransparentops),
    (name="test_flatten_axis1", ninputs=1, noutputs=1, fd=pseudotransparentops),
    (name="test_flatten_axis2", ninputs=1, noutputs=1, fd=pseudotransparentops),
    (name="test_flatten_axis3", ninputs=1, noutputs=1, fd=pseudotransparentops),
    (name="test_flatten_default_axis", ninputs=1, noutputs=1, fd=pseudotransparentops),
    (name="test_flatten_negative_axis1", ninputs=1, noutputs=1, fd=pseudotransparentops),
    (name="test_flatten_negative_axis2", ninputs=1, noutputs=1, fd=pseudotransparentops),
    (name="test_flatten_negative_axis3", ninputs=1, noutputs=1, fd=pseudotransparentops),
    (name="test_globalaveragepool", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_globalaveragepool_precomputed", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_globalmaxpool", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_globalmaxpool_precomputed", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reduce_mean_default_axes_keepdims_example", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reduce_mean_default_axes_keepdims_random", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reduce_mean_do_not_keepdims_example", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reduce_mean_do_not_keepdims_random", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reduce_mean_keepdims_example", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reduce_mean_keepdims_random", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reduce_mean_negative_axes_keepdims_example", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reduce_mean_negative_axes_keepdims_random", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_reshape_extended_dims", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_reshape_negative_dim", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_reshape_negative_extended_dims", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_reshape_one_dim", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_reshape_reduced_dims", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_reshape_reordered_all_dims", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_reshape_reordered_last_dims", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_reshape_zero_and_negative_dim", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_reshape_zero_dim", ninputs=2, noutputs=1, fd=pseudotransparentops),
    (name="test_softmax_axis_0", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_softmax_axis_1", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_softmax_axis_2", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_softmax_default_axis", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_softmax_example", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_softmax_large_number", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_softmax_negative_axis", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_squeeze", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_squeeze_negative_axes", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_unsqueeze_axis_0", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_unsqueeze_axis_1", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_unsqueeze_axis_2", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_unsqueeze_axis_3", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_unsqueeze_negative_axes", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_unsqueeze_three_axes", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_unsqueeze_two_axes", ninputs=1, noutputs=1, fd=invariantops),
    (name="test_unsqueeze_unsorted_axes", ninputs=1, noutputs=1, fd=invariantops),
    )

    model, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) op $(optype(node))" for node in nodes(gb)
        @test haskey(tc.fd, optype(node))
        op = tc.fd[optype(node)](node.attribute, params(node)...)
        res = op(inputs[1])
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end

    @testset "$(tc.name) graph" begin
        cg = load(model, size(inputs[1]))
        res = cg(inputs[1])
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]

        # Also test that it we get the same thing by serializing and then deserializing
        io = PipeBuffer()
        save(io, cg)
        cg = load(io, size(inputs[1]))
        res = cg(inputs[1])
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end
end

@testset "Vertex $(tc.name)" for tc in
    (
    (name="test_add", ninputs=2, noutputs=1),
    (name="test_div", ninputs=2, noutputs=1),
    #(name="test_add_bcast", ninputs=2, noutputs=1), # Op is supported, but we get the wrong idea about what type of inputvertex to create from 3D input
    (name="test_concat_1d_axis_0", ninputs=2, noutputs=1),
    (name="test_concat_1d_axis_negative_1", ninputs=2, noutputs=1),
    (name="test_concat_2d_axis_0", ninputs=2, noutputs=1),
    (name="test_concat_2d_axis_1", ninputs=2, noutputs=1),
    (name="test_concat_2d_axis_negative_1", ninputs=2, noutputs=1),
    (name="test_concat_2d_axis_negative_2", ninputs=2, noutputs=1),
    (name="test_concat_3d_axis_0", ninputs=2, noutputs=1),
    (name="test_concat_3d_axis_1", ninputs=2, noutputs=1),
    (name="test_concat_3d_axis_2", ninputs=2, noutputs=1),
    (name="test_concat_3d_axis_negative_1", ninputs=2, noutputs=1),
    (name="test_concat_3d_axis_negative_2", ninputs=2, noutputs=1),
    (name="test_concat_3d_axis_negative_3", ninputs=2, noutputs=1),
    (name="test_matmul_2d", ninputs=2, noutputs=1),
    (name="test_mul", ninputs=2, noutputs=1),
    )

    model, gb, ins, outs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) graph" begin
        cg = load(model)
        res = cg(ins[1:length(inputs(cg))]...)
        @test size(res) == size(outs[1])
        @test res ≈ outs[1]

        # Also test that it we get the same thing by serializing and then deserializing
        io = PipeBuffer()
        save(io, cg)
        cg = load(io)
        res = cg(ins[1:length(inputs(cg))]...)
        @test size(res) == size(outs[1])
        @test res ≈ outs[1]
    end
end

@testset "Deserialize with inputs" begin
    using NaiveNASflux: GenericFlux2D, layertype

    function sumgraph()
        ivs = denseinputvertex.(["in1", "in2"], 4) 
        g_org = CompGraph(ivs, "out" >> ivs[1] + ivs[2])
        pb = PipeBuffer()
        save(pb, g_org, "in1" => missing, "in2" => missing)
        return pb
    end

    insize(t::Tuple) = ONNXNaiveNASflux.int_size(t[NaiveNASflux.actdim(length(t))])
    insize(p::Pair) = p |> last |> insize
    @testset "Input format $inshapes" for inshapes in (
        ((4,1), (4,1)),
        ("in1" => (4,1), "in2" => (4,1)),
        ((4,missing), (4, :B)),
        ((:I, 3), (:I, 4))
    ) 
        expsizes =  insize.(inshapes) |> collect
        g_new = if any(==(0), expsizes)
            @test_logs (:warn, r"No valid input sizes") load(sumgraph(), inshapes...)
        else
            load(sumgraph(), inshapes...)
        end
        @test nout.(inputs(g_new)) == expsizes
        @test layertype.(inputs(g_new)) == [GenericFlux2D(), GenericFlux2D()]
    end

    inshape(t::Tuple) = t |> length |> ONNXNaiveNASflux.guess_layertype
    @testset "Mixshape format $inshapes" for inshapes in (
        ((1,1,5,1), (5,1)),
        ((5,1), (1,1,5,1)),
    )
        g_new = load(sumgraph(), inshapes...)
        @test nout.(inputs(g_new)) == [5, 5]
        @test layertype.(inputs(g_new)) == inshape.(inshapes |> collect)
    end

    @testset "Malformed input $inshapes" for inshapes in (
        ((4,1), (4,1), (4,1)),
        ("in1" => (4,1), "in2" => (4,1), "in2" => (4,1)),
        ("in1" => (4,1), "notin2" => (4,1))
    )
        @test_throws AssertionError load(sumgraph(), inshapes...)
    end
end

@testset "Deserialize with merging" begin
    function remodel(f, args...) 
        pb = PipeBuffer()
        save(pb, f, args...)
        return load(pb, args...)
    end


    @testset "Merge activation function" begin
        m = remodel(Dense(3,4, relu), (3, missing))
        @test nvertices(m) == 2
        @test layer(outputs(m)[1]).σ == relu
    end

    @testset "Merge Reshape and $gp" for gp in (
        ONNXNaiveNASflux.globalmeanpool,
        ONNXNaiveNASflux.globalmaxpool
    ) 
        m = remodel(Chain(
            Conv((3,3), 3 => 3),
            x -> gp(x, xf -> reshape(xf, 3, :)),
        ), (4, 4, 3, missing))

        @test nvertices(m) == 3
    end

    @testset "Merge constant" begin
        m = remodel(x -> relu.(x) .+ (3))
        @test nvertices(m) == 3
        @test m([-1,1]) == [3, 4] 
    end
end

@testset "Infer CompGraph shapes" begin
    function remodel(f, args...;kwargs...) 
        pb = PipeBuffer()
        save(pb, f, args...)
        return load(pb,; kwargs...)
    end

    @testset "Simple Dense" begin
        v1 = denseinputvertex("v1", 3)
        v2 = fluxvertex("v2", Dense(nout(v1), 4), v1)
        g = remodel(CompGraph(v1, v2), missing)
        
        @test nout(inputs(g)[]) == nout(v1)
    end

    
    @testset "After invariant" begin
        v1 = denseinputvertex("v1", 3)
        v2 = invariantvertex("v2", identity, v1)
        v3 = fluxvertex("v3", Dense(nout(v2), 4), v2)
        g = remodel(CompGraph(v1, v3), missing)

        @test nout(inputs(g)[]) == nout(v1)
    end

    @testset "Can't infer after concat" begin
        # I suppose in this case we could infer it as we know nin(v3) = 2 * nout(v1)
        # Seems like too much of an edge case to be worth considering though
        v1 = denseinputvertex("v1", 3)
        v2 = concat("v2", v1, v1)
        v3 = fluxvertex("v3", Dense(nout(v2), 4), v2)
        g = @test_logs (:warn, r"No valid input sizes") remodel(CompGraph(v1, v3), (missing, :B))

        @test nout(inputs(g)[]) == 0
    end

    @testset "Flatten$label" for (label, layerfun, exputilsize) in 
        (
        ("", identity, 1),
        (" with ActivationContribution", ActivationContribution, 60),
        )
        using ONNXNaiveNASflux: Flatten, create_vertex_default, defaultutility

        v1 = conv2dinputvertex("v1", 3)
        v2 = fluxvertex("v2", Conv((2,2), 3=>5), v1)
        v3 = absorbvertex("v3", Flatten(-1), v2)

        g = remodel(CompGraph(v1, v3), (5,4,3,:B); vfun=(args...) -> create_vertex_default(args...; layerfun))

        @test nout(inputs(g)[]) == nout(v1)
        @test nout(g[end]) == 60
        @test length(defaultutility(g[end])) == exputilsize
    end

    @testset "Squeeze" begin
        import ONNXNaiveNASflux: Squeeze, NumPyAxes

        # The top two are not hit through any other test
        @test size(Squeeze((1,2))(reshape(ones(2,3), 1, 1, 2, 1, 3, 1))) == (2, 1, 3, 1)
        @test size(Squeeze(missing)(reshape(ones(2,3), 1, 1, 2, 1, 3, 1))) == (2, 3)
        @test size(Squeeze(NumPyAxes([-1, 0, 2]))(reshape(ones(2,3), 1, 1, 2, 1, 3, 1))) == (1, 2, 3)
    end

    @testset "Pretty printing" begin
        import ONNXNaiveNASflux: NumPyAxes

        @testset "NumPyAxes" begin
            @test sprint(show, NumPyAxes([0, 1, 2, -1, 2])) == "NumPyAxes[end,end-1,end-2,1,end-2]"           
        end

        @testset "Squeeze" begin
            import ONNXNaiveNASflux: Squeeze
            @test sprint(show, Squeeze(NumPyAxes([0, 1, -2]))) == "Squeeze(dims=[end,end-1,2])"
            @test sprint(show, Squeeze(missing)) == "Squeeze" 
        end

        @testset "Unsqueeze" begin
            import ONNXNaiveNASflux: Unsqueeze
            @test sprint(show, Unsqueeze(NumPyAxes([0, 1, -2]))) == "Unsqueeze(dims=[end,end-1,2])"
        end

        @testset "Reshape" begin
            import ONNXNaiveNASflux: Reshape
            @test sprint(show, Reshape((1, 2))) == "Reshape(dims=(1, 2))"
        end

        @testset "Flatten" begin
            import ONNXNaiveNASflux: Flatten
            @test sprint(show, Flatten(2)) == "Flatten(dim=2)"
        end
    end
end