import ONNXmutable: protos, optype, actfuns, fluxlayers
using NaiveNASflux
import NaiveNASflux: weights, bias

function serdeser(p, pt, convfun = cfun(pt))
    iob = PipeBuffer();
    ONNX.writeproto(iob, p)
    return convfun(ONNX.readproto(iob, pt))
end

cfun(pt) = ONNX.convert
cfun(pt::ONNX.Proto.TensorProto) = ONNX.get_array

@testset "Tensor shape $s" for s in ((1,), (2,), (3,4), (1,2,3), (4,5,6,7))
    exp = reshape(collect(Float32, 1:prod(s)), s)
    @show size(exp)

    tp = ONNX.Proto.TensorProto(exp, "testtensor")

    res = serdeser(tp, ONNX.Proto.TensorProto())

    @test size(res) == size(exp)
    @test eltype(res) == eltype(exp)
    @test res == exp

end

@testset "Paramless function $(tc.f)" for tc in (
    (f=relu, ot="Relu")
    ,)

    inname = ["input"]
    outname = "output"

    np = protos(tc.f, inname, t -> outname)[1]
    res = serdeser(np, ONNX.Proto.NodeProto())

    @test res.input == inname
    @test res.output == [outname * "_fwd"]
    @test res.op_type == tc.ot
    @test res.name == outname * "_fwd"

end

@testset "Dense layer actfun $af" for af in (
    relu,
    )
    exp = Dense(3,4, af)

    inname = ["input"]

    dp,ap,wp,bp = protos(exp, inname, l -> lowercase(string(typeof(l))))

    dn = serdeser(dp, ONNX.Proto.NodeProto())
    an = serdeser(ap, ONNX.Proto.NodeProto())
    w = serdeser(wp, ONNX.Proto.TensorProto())
    b = serdeser(bp, ONNX.Proto.TensorProto())

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
