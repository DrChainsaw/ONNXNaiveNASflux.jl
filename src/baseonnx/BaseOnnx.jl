module BaseOnnx

  include("onnx3_pb.jl")

  import ProtoBuf

  const TensorProto_Segment = var"TensorProto.Segment"
  const TensorProto_DataType = var"TensorProto.DataType"
  const TensorProto_DataLocation = var"TensorProto.DataLocation"
  const TensorShapeProto_Dimension = var"TensorShapeProto.Dimension"
  const AttributeProto_AttributeType = var"AttributeProto.AttributeType"
  const TypeProto_SparseTensor = var"TypeProto.SparseTensor"
  const TypeProto_Tensor = var"TypeProto.Tensor"
  const TypeProto_Sequence = var"TypeProto.Sequence"
  const TypeProto_Optional = var"TypeProto.Optional"
  const TypeProto_Map = var"TypeProto.Map"

  include("read.jl")
  include("write.jl")
end
