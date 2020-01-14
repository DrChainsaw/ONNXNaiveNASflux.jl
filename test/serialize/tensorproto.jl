

@testset "TensorProto" begin

    using NaiveNASflux
    import NaiveNASflux: actrank

    function serdeser(p::T, convfun = cfun(p)) where T
        iob = PipeBuffer();
        ONNX.writeproto(iob, p)
        return convfun(ONNX.readproto(iob, T()))
    end

    cfun(::ONNX.Proto.TensorProto) = ONNX.get_array
    cfun(::ONNX.Proto.ValueInfoProto) = vip -> (ONNXmutable.name(vip), ONNXmutable.size(vip))

    @testset "Tensor shape $s" for s in ((1,), (2,), (3,4), (1,2,3), (4,5,6,7))
        exp = reshape(collect(Float32, 1:prod(s)), s)

        tp = ONNX.Proto.TensorProto(exp, "testtensor")

        res = serdeser(tp)

        @test size(res) == size(exp)
        @test eltype(res) == eltype(exp)
        @test res == exp

    end

    @testset "ValueInfo shape $s" for s in (FluxDense(), FluxConv{1}(), FluxConv{2}(), FluxConv{3}())

        tp = ONNX.Proto.ValueInfoProto("test", (3, s))

        name,vsize = serdeser(tp)

        @test name == "test"
        @test length(vsize) == 1 + actrank(s)
        @test vsize[end-1] == 3
    end
end
