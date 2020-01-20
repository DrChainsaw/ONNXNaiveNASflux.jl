

ONNX.Proto.ValueInfoProto(name::String, inshape) =
ONNX.Proto.ValueInfoProto(
    name=name,
    _type=ONNX.Proto.TypeProto(
        tensor_type=ONNX.Proto.TypeProto_Tensor(
            shape=ONNX.Proto.TensorShapeProto(inshape)
        )
    )
)
ONNX.Proto.TensorShapeProto(shape) = ONNX.Proto.TensorShapeProto(dim=[tsp_d(s) for s in shape])
ONNX.Proto.TensorShapeProto(::Missing) = ONNX.Proto.TensorShapeProto()
tsp_d(::Missing) = ONNX.Proto.TensorShapeProto_Dimension()
tsp_d(n::Integer) = ONNX.Proto.TensorShapeProto_Dimension(dim_value=n)

ONNX.Proto.TensorProto(t::AbstractArray{Float32,N}, name ="") where N = ONNX.Proto.TensorProto(
    dims=collect(reverse(size(t))),
    data_type=ONNX.Proto.TensorProto_DataType.FLOAT,
    float_data = reshape(t,:),
    name=name)

ONNX.Proto.AttributeProto(name::String, i::Int64) = ONNX.Proto.AttributeProto(
    name=name,
    _type = ONNX.Proto.AttributeProto_AttributeType.INT,
    i = i
)

ONNX.Proto.AttributeProto(name::String, f::Float32) = ONNX.Proto.AttributeProto(
    name=name,
    _type = ONNX.Proto.AttributeProto_AttributeType.FLOAT,
    f = f
)

ONNX.Proto.AttributeProto(name::String, f::Float64) = ONNX.Proto.AttributeProto(
    name=name,
    _type = ONNX.Proto.AttributeProto_AttributeType.FLOAT,
    f = Float32(f)
)

ONNX.Proto.AttributeProto(name::String, i::NTuple{N, Int64}) where N = ONNX.Proto.AttributeProto(
    name=name,
    _type = ONNX.Proto.AttributeProto_AttributeType.INTS,
    ints = collect(i)
)
