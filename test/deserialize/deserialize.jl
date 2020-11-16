import ONNXmutable: fluxlayers, sources, actfuns, invariantops, pseudotransparentops, optype, nodes
using NaiveNASflux

# Logging to avoid travis timeouts
@info "  Test padding and sources"

@testset "Read padding" begin
    import ONNXmutable: prev

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
        cg = CompGraph(model)
        res = cg()
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]

        # Also test that it we get the same thing by serializing and then deserializing
        io = PipeBuffer()
        onnx(io, cg)
        cg = CompGraph(io)
        res = cg()
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end
end

# For testing since ONNX states that recurrent layers take 3D input while flux uses
# an Array of 2D Arrays
function (l::Flux.Recur)(x::AbstractArray{T, 3}) where T
    # ONNX shape for RNNs inputs is [seq_length, batch_size, input_size]
    # ONNX.jl reverses this to [input_size, batch_size, seq_length]
    # Unstacking it to a sequence of [input_size, batch_size]
    inseq =Flux.unstack(x, 3)
    out = nothing
    for inpt in inseq
         out = l(inpt)
     end
    # Just to turn it back to ONNX shape.
    # In the testdata only the last output in the sequence is present in the reference
    return reshape(out, size(out)..., 1)
end

@info "  Test Flux layers"

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
    (name="test_dropout_default", ninputs=1, noutputs=1),
    (name="test_dropout_random", ninputs=1, noutputs=1),
    #(name="test_gemm_all_attributes", ninputs=3, noutputs=1), Not supported!
    (name="test_gemm_alpha", ninputs=3, noutputs=1),
    (name="test_gemm_beta", ninputs=3, noutputs=1),
    (name="test_gemm_default_matrix_bias", ninputs=3, noutputs=1),
    (name="test_gemm_default_no_bias", ninputs=2, noutputs=1),
    (name="test_gemm_default_scalar_bias", ninputs=3, noutputs=1),
    (name="test_gemm_default_single_elem_vector_bias", ninputs=3, noutputs=1),
    (name="test_gemm_default_vector_bias", ninputs=3, noutputs=1),
    (name="test_gemm_default_zero_bias", ninputs=3, noutputs=1),
    #(name="test_gemm_transposeA", ninputs=3, noutputs=1), Not supported!
    (name="test_gemm_transposeB", ninputs=3, noutputs=1),
    (name="test_lstm_defaults", ninputs=3, noutputs=1),
    (name="test_lstm_with_initial_bias", ninputs=4, noutputs=1),
    # (name="test_lstm_with_peepholes", ninputs=8, noutputs=1), Not supported!
    (name="test_maxpool_1d_default", ninputs=1, noutputs=1),
    #(name="test_maxpool_2d_ceil", ninputs=1, noutputs=1), Not supported!
    (name="test_maxpool_2d_default", ninputs=1, noutputs=1),
    #(name="test_maxpool_2d_dilations", ninputs=1, noutputs=1), Not supported!
    #(name="test_maxpool_2d_pads", ninputs=1, noutputs=1), Not supported!
    (name="test_maxpool_2d_strides", ninputs=1, noutputs=1),
    (name="test_maxpool_3d_default", ninputs=1, noutputs=1),
    (name="test_maxpool_3d_default", ninputs=1, noutputs=1),
    (name="test_rnn_seq_length", ninputs=4, noutputs=1),
    )

    model, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) op $(optype(node))" for node in nodes(gb)
        @test haskey(fluxlayers, optype(node))
        op = fluxlayers[optype(node)](node.attribute, params(node)...)

        res = op(Float32.(inputs[1]))
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end

    @testset "$(tc.name) graph" begin
        cg = CompGraph(model)
        res = cg(Float32.(inputs[1]))
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]

        # Also test that it we get the same thing by serializing and then deserializing
        io = PipeBuffer()
        onnx(io, cg)
        cg = CompGraph(io)
        res = cg(Float32.(inputs[1]))
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
    (name="test_selu", ninputs=1, noutputs=1),
    (name="test_selu_default", ninputs=1, noutputs=1),
    (name="test_selu_example", ninputs=1, noutputs=1),
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
    (name="test_squeeze_negative_axes", ninputs=1, noutputs=1, fd=invariantops)
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
        cg = CompGraph(model)
        res = cg(inputs[1])
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]

        # Also test that it we get the same thing by serializing and then deserializing
        io = PipeBuffer()
        onnx(io, cg)
        cg = CompGraph(io)
        res = cg(inputs[1])
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end
end

@testset "Vertex $(tc.name)" for tc in
    (
    (name="test_add", ninputs=2, noutputs=1),
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
    (name="test_mul", ninputs=2, noutputs=1),
    )

    model, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) graph" begin
        cg = CompGraph(model)
        res = cg(inputs[1:length(cg.inputs)]...)
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]

        # Also test that it we get the same thing by serializing and then deserializing
        io = PipeBuffer()
        onnx(io, cg)
        cg = CompGraph(io)
        res = cg(inputs[1:length(cg.inputs)]...)
        @test size(res) == size(outputs[1])
        @test res ≈ outputs[1]
    end
end

@testset "Deserialize with inputs" begin

    function sumgraph()
        ivs = inputvertex.(["in1", "in2"], 4, Ref(FluxDense())) 
        g_org = CompGraph(ivs, "out" >> ivs[1] + ivs[2])
        pb = PipeBuffer()
        onnx(pb, g_org, "in1" => missing, "in2" => missing)
        return pb
    end

    insize(t::Tuple) = ONNXmutable.clean_size(t[NaiveNASflux.actdim(length(t))])
    insize(p::Pair) = p |> last |> insize
    @testset "Input format $inshapes" for inshapes in (
        ((4,1), (4,1)),
        ("in1" => (4,1), "in2" => (4,1)),
        ((4,missing), (4, :B)),
        ((:I, 3), (:I, 4))
    ) 
        g_new = CompGraph(sumgraph(), inshapes...)
        @test nout.(g_new.inputs) == insize.(inshapes) |> collect
        @test layertype.(g_new.inputs) == [FluxDense(), FluxDense()]
    end

    inshape(t::Tuple) = t |> length |> ONNXmutable.guess_layertype
    @testset "Mixshape format $inshapes" for inshapes in (
        ((1,1,5,1), (5,1)),
        ((5,1), (1,1,5,1)),
    )
        g_new = CompGraph(sumgraph(), inshapes...)
        @test nout.(g_new.inputs) == [5, 5]
        @test layertype.(g_new.inputs) == inshape.(inshapes |> collect)
    end

    @testset "Malformed input $inshapes" for inshapes in (
        ((4,1), (4,1), (4,1)),
        ("in1" => (4,1), "in2" => (4,1), "in2" => (4,1)),
        ("in1" => (4,1), "notin2" => (4,1))
    )
        @test_throws AssertionError CompGraph(sumgraph(), inshapes...)
    end
end

@testset "Deserialize with merging" begin
    function remodel(f, args...) 
        pb = PipeBuffer()
        onnx(pb, f, args...)
        return CompGraph(pb, args...)
    end


    @testset "Merge activation function" begin
        m = remodel(Dense(3,4, relu), (3, missing))
        @test nv(m) == 2
        @test layer(m.outputs[1]).σ == relu
    end

    @testset "Merge Reshape and $gp" for gp in (
        ONNXmutable.globalmeanpool,
        ONNXmutable.globalmaxpool
    ) 
        m = remodel(Chain(
            Conv((3,3), 3 => 3),
            x -> gp(x, xf -> reshape(xf, 3, :)),
        ), (4, 4, 3, missing))

        @test nv(m) == 3
    end

    @testset "Merge constant" begin
        m = remodel(x -> relu.(x) .+ (3))
        @test nv(m) == 3
        @test m([-1,1]) == [3, 4] 
    end
end