module ONNXNaiveNASflux

include("baseonnx/BaseOnnx.jl")

import .BaseOnnx: array
const ONNX = BaseOnnx
using Flux
using Flux: params
using NaiveNASflux
using NaiveNASflux: weights, bias
using NaiveNASflux: indim, outdim, actdim, actrank, layertype, wrapped
using NaiveNASflux: FluxLayer, FluxParLayer, FluxNoParLayer, FluxDense, FluxConvolutional, FluxConv, FluxBatchNorm, 
                    FluxRecurrent, FluxRnn, FluxLstm, FluxGru, FluxTransparentLayer, FluxPoolLayer, FluxDropOut, Flux2D,
                    GenericFluxConvolutional, GenericFlux2D, GenericFluxRecurrent
using Setfield
using Statistics
import Pkg
import JuMP: @variable, @constraint
using NaiveNASlib.Extend, NaiveNASlib.Advanced
using NaiveNASlib: compconstraint!, all_in_Δsize_graph, NamedTrait, VertexConf

export load, save

include("shapes.jl")
include("validate.jl")

include("deserialize/vertex.jl")
include("deserialize/infershape.jl")
include("deserialize/constraints.jl")
include("deserialize/ops.jl")
include("deserialize/graphbuilder.jl")
include("deserialize/combine.jl")
include("deserialize/deserialize.jl")

include("serialize/namingutil.jl")
include("serialize/serialize.jl")

end # module
