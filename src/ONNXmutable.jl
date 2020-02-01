module ONNXmutable

import ONNX
import ONNX: readproto, convert, Types, Proto
using NaiveNASflux
import NaiveNASflux: weights, bias
import NaiveNASflux: indim, outdim, actdim, actrank
using Setfield
using Statistics
import Pkg

export onnx

include("shapes.jl")
include("validate.jl")

include("deserialize/ops.jl")
include("deserialize/graphbuilder.jl")
include("deserialize/deserialize.jl")

include("serialize/namingutil.jl")
include("serialize/protos.jl")
include("serialize/serialize.jl")

end # module
