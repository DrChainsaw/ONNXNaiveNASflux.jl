

ONNX.Proto.ValueInfoProto(name::String, inshape) =
ONNX.Proto.ValueInfoProto(
    name=name,
    _type=ONNX.Proto.TypeProto(
        tensor_type=ONNX.Proto.TypeProto_Tensor(
            shape=ONNX.Proto.TensorShapeProto(
                dim=[tsp_d(), Iterators.flatten((tsp_d(s) for s in inshape))...]
            )
        )
    )
)

tsp_d() = ONNX.Proto.TensorShapeProto_Dimension()
tsp_d(n::Integer) = (ONNX.Proto.TensorShapeProto_Dimension(dim_value=n),)
tsp_d(l::FluxLayer) = (tsp_d() for i in 1:NaiveNASflux.actrank(l)-1)
tsp_d(l::FluxRecurrent) = (tsp_d(),) # Flux uses a sequence of 2D input to recurrent layers

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

ONNX.Proto.AttributeProto(name::String, i::NTuple{N, Int64}) where N = ONNX.Proto.AttributeProto(
    name=name,
    _type = ONNX.Proto.AttributeProto_AttributeType.INTS,
    ints = collect(i)
)
