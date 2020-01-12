module ONNXmutable

import ONNX
import ONNX: readproto, convert, Types, Proto
using NaiveNASflux
import NaiveNASflux: weights, bias
using Setfield

include("deserialize/ops.jl")
include("deserialize/deserialize.jl")

include("serialize/serialize.jl")

end # module
