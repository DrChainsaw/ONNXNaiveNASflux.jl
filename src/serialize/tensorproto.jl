
# TODO: User supplied elemtype??
ONNX.Proto.ValueInfoProto(name::String, inshape, elemtype=Float32) =
ONNX.Proto.ValueInfoProto(
    name=name,
    _type=ONNX.Proto.TypeProto(
        tensor_type=ONNX.Proto.TypeProto_Tensor(
            elem_type=tp_tensor_elemtype(elemtype),
            shape=ONNX.Proto.TensorShapeProto(inshape)
        )
    )
)
ONNX.Proto.TensorShapeProto(shape) = ONNX.Proto.TensorShapeProto(dim=[tsp_d(s) for s in reverse(shape)])
ONNX.Proto.TensorShapeProto(::Missing) = ONNX.Proto.TensorShapeProto()
tsp_d(::Missing) = ONNX.Proto.TensorShapeProto_Dimension()
tsp_d(n::Integer) = ONNX.Proto.TensorShapeProto_Dimension(dim_value=n)
tsp_d(s::String) = ONNX.Proto.TensorShapeProto_Dimension(dim_param=s)
tsp_d(s::Symbol) = tsp_d(string(s))

tp_tensor_elemtype(::Missing) = ONNX.Proto.TensorProto_DataType.UNDEFINED
tp_tensor_elemtype(::Type{Float32}) = ONNX.Proto.TensorProto_DataType.FLOAT

ONNX.Proto.TensorProto(t::AbstractArray{Float32,N}, name ="") where N = ONNX.Proto.TensorProto(
    dims=collect(reverse(size(t))),
    data_type=ONNX.Proto.TensorProto_DataType.FLOAT,
    float_data = reshape(t,:),
    name=name)

ONNX.Proto.TensorProto(t::AbstractArray{Int64,N}, name ="") where N = ONNX.Proto.TensorProto(
    dims=collect(reverse(size(t))),
    data_type=ONNX.Proto.TensorProto_DataType.INT64,
    int64_data = reshape(t,:),
    name=name)

ONNX.Proto.TensorProto(t::AbstractArray{Int32,N}, name ="") where N = ONNX.Proto.TensorProto(
    dims=collect(reverse(size(t))),
    data_type=ONNX.Proto.TensorProto_DataType.INT32,
    int32_data = reshape(t,:),
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

ONNX.Proto.AttributeProto(name::String, i::NTuple{N, Int64}) where N = ONNX.Proto.AttributeProto(name, collect(i))

ONNX.Proto.AttributeProto(name::String, i::AbstractVector{Int64}) where N = ONNX.Proto.AttributeProto(
    name=name,
    _type = ONNX.Proto.AttributeProto_AttributeType.INTS,
    ints = i
)

ONNX.Proto.AttributeProto(name::String, s::String) where N = ONNX.Proto.AttributeProto(
    name=name,
    _type = ONNX.Proto.AttributeProto_AttributeType.STRING,
    s = Vector{UInt8}(s)
)
