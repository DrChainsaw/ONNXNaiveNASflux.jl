module ONNXmutable

include("baseonnx/BaseOnnx.jl")

import .BaseOnnx: array
const ONNX = BaseOnnx
using NaiveNASflux
import NaiveNASflux: weights, bias
import NaiveNASflux: indim, outdim, actdim, actrank
using Setfield
using Statistics
import Pkg
import JuMP: @variable, @constraint
import NaiveNASflux.NaiveNASlib: compconstraint!, all_in_Δsize_graph

export onnx, CompGraph

include("shapes.jl")
include("validate.jl")

include("deserialize/vertex.jl")
include("deserialize/constraints.jl")
include("deserialize/ops.jl")
include("deserialize/graphbuilder.jl")
include("deserialize/combine.jl")
include("deserialize/deserialize.jl")

include("serialize/namingutil.jl")
include("serialize/serialize.jl")

end # module
