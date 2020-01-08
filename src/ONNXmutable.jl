module ONNXmutable

import ONNX
import ONNX: readproto, convert, Types, Proto
using NaiveNASflux
using Setfield

include("deserialize/ops.jl")
include("deserialize/deserialize.jl")

end # module
