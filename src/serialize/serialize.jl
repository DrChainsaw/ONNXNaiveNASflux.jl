

function protos(l::Dense, innames, namestrat)
    @assert length(innames) == 1 "Only one input expected! Got $innames"

    lname = namestrat(l)
    oname, wname, bname = lname .* ("_fwd", "_weight", "_bias")

    return ONNX.Proto.NodeProto(
        input=[innames[], wname, bname],
        output=[oname],
        name=oname,
        op_type="Gemm"),
    protos(l.σ, [oname], f -> join([lname, lowercase(string(f))], "_"))...,
    ONNX.Proto.TensorProto(weights(l), wname),
    ONNX.Proto.TensorProto(bias(l), bname)
end


protos(::typeof(identity), innames, namestrat) = ()

function protos(::typeof(relu), innames, namestrat)
    oname = namestrat(relu) * "_fwd"
    return (ONNX.Proto.NodeProto(
    input = innames,
    output = [oname],
    name=oname,
    op_type="Relu"), )
end


ONNX.Proto.TensorProto(t::AbstractArray{Float32,N}, name ="") where N = ONNX.Proto.TensorProto(
    dims=collect(reverse(size(t))),
    data_type=ONNX.Proto.TensorProto_DataType.FLOAT,
    float_data = reshape(t,:),
    name=name)
