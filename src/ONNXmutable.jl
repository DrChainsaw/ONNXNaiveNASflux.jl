module ONNXmutable

import ONNX
import ONNX: readproto, convert, Types, Proto
using NaiveNASflux
import NaiveNASflux: weights, bias
import NaiveNASflux: indim, outdim, actdim, actrank
using Setfield
using Statistics
import Pkg

include("shapes.jl")


include("deserialize/ops.jl")
include("deserialize/graphbuilder.jl")
include("deserialize/deserialize.jl")

include("serialize/namingutil.jl")
include("serialize/tensorproto.jl")
include("serialize/serialize.jl")

end # module
