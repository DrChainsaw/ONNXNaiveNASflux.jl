import ONNXmutable: fluxlayers, actfuns, invariantops, optype, params
import NaiveNASflux: CompGraph

@testset "Fluxlayer $(tc.name)" for tc in
    (
    (name="test_averagepool_1d_default", ninputs=1, noutputs=1),
    #(name="test_averagepool_2d_ceil", ninputs=1, noutputs=1), Not supported!
    (name="test_averagepool_2d_default", ninputs=1, noutputs=1),
    #(name="test_averagepool_2d_pads", ninputs=1, noutputs=1), Not supported!
    (name="test_averagepool_2d_strides", ninputs=1, noutputs=1),
    (name="test_averagepool_3d_default", ninputs=1, noutputs=1),
    (name="test_basic_conv_with_padding", ninputs=2, noutputs=1),
    (name="test_basic_conv_without_padding", ninputs=2, noutputs=1),
    (name="test_batchnorm_epsilon", ninputs=5, noutputs=1),
    (name="test_batchnorm_example", ninputs=5, noutputs=1),
    #(name="test_conv_with_strides_and_asymmetric_padding", ninputs=2, noutputs=1), Not supported!
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
    (name="test_maxpool_1d_default", ninputs=1, noutputs=1),
    #(name="test_maxpool_2d_ceil", ninputs=1, noutputs=1), Not supported!
    (name="test_maxpool_2d_default", ninputs=1, noutputs=1),
    #(name="test_maxpool_2d_dilations", ninputs=1, noutputs=1), Not supported!
    #(name="test_maxpool_2d_pads", ninputs=1, noutputs=1), Not supported!
    (name="test_maxpool_2d_strides", ninputs=1, noutputs=1),
    (name="test_maxpool_3d_default", ninputs=1, noutputs=1),
    (name="test_maxpool_3d_default", ninputs=1, noutputs=1))

    model, sizes, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) op $(node.op_type)" for node in gb.g.node
        @test haskey(fluxlayers, optype(node))
        op = fluxlayers[optype(node)](node.attribute, params(node, gb)...)
        @test op(inputs[1]) ≈ outputs[1]
    end

    @testset "$(tc.name) graph" begin
        cg = CompGraph(model, sizes)
        @test cg(inputs[1]) ≈ outputs[1]
    end

end

@testset "Activation functions $(tc.name)" for tc in
    (
    (name="test_relu", ninputs=1, noutputs=1),)

    model, sizes, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) op $(node.op_type)" for node in gb.g.node
        @test haskey(actfuns, optype(node))
        op = actfuns[optype(node)](node.attribute, params(node, gb)...)
        @test op.(inputs[1]) ≈ outputs[1]
    end
end

@testset "Invariant op $(tc.name)" for tc in
    (
    (name="test_globalaveragepool", ninputs=1, noutputs=1),
    (name="test_globalaveragepool_precomputed", ninputs=1, noutputs=1),
    (name="test_reshape_extended_dims", ninputs=2, noutputs=1),
    (name="test_reshape_negative_dim", ninputs=2, noutputs=1),
    (name="test_reshape_negative_extended_dims", ninputs=2, noutputs=1),
    (name="test_reshape_one_dim", ninputs=2, noutputs=1),
    (name="test_reshape_reduced_dims", ninputs=2, noutputs=1),
    (name="test_reshape_reordered_all_dims", ninputs=2, noutputs=1),
    (name="test_reshape_reordered_last_dims", ninputs=2, noutputs=1),
    (name="test_reshape_zero_and_negative_dim", ninputs=2, noutputs=1),
    (name="test_reshape_zero_dim", ninputs=2, noutputs=1))

    model, sizes, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) op $(node.op_type)" for node in gb.g.node
        @test haskey(invariantops, optype(node))
        op = invariantops[optype(node)](node.attribute, params(node, gb)...)
        @test op(inputs[1]) ≈ outputs[1]
    end

    @testset "$(tc.name) graph" begin
        cg = CompGraph(model, sizes)
        @test cg(inputs[1]) ≈ outputs[1]
    end
end

@testset "Vertex $(tc.name)" for tc in
    (
    (name="test_add", ninputs=2, noutputs=1),
    #(name="test_add_bcast", ninputs=2, noutputs=1) # Op is supported, but we get the wrong idea about what type of inputvertex to create from 3D input
    )

    model, sizes, gb, inputs, outputs = prepare_node_test(tc.name, tc.ninputs, tc.noutputs)

    @testset "$(tc.name) graph" begin
        cg = CompGraph(model, sizes)
        @test cg(inputs[1:length(cg.inputs)]...) ≈ outputs[1]
    end
end
