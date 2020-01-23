# ONNXmutable

[![Build Status](https://travis-ci.com/DrChainsaw/ONNXmutable.jl.svg?branch=master)](https://travis-ci.com/DrChainsaw/ONNXmutable.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/ONNXmutable.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/ONNXmutable-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/ONNXmutable.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/ONNXmutable.jl)

ONNXmutable is an extension of [ONNX.jl](https://github.com/FluxML/ONNX.jl) which adds serialization of (almost) arbitrary functions into [ONNX](https://onnx.ai) models.

It is also capable of deserializing models into [NaiveNASflux](https://github.com/DrChainsaw/NaiveNASflux.jl) graphs, bringing its powerful mutation capabilities to the transfer learning context.

## Basic usage

```julia
Pkg.add("https://github.com/DrChainsaw/ONNXmutable.jl")
```

Serialization is done using the `onnx` function which accepts a filename `String` or an `IO` as first argument:

```julia
using ONNXmutable, NaiveNASflux, Test

l1 = Conv((3,3), 2=>3, relu)
l2 = Dense(3, 4, elu)

f = function(x,y)
    x = l1(x)
    # Home-brewed global average pool
    x = dropdims(mean(x, dims=(1,2)), dims=(1,2))
    x = l2(x)
    return x + y
end

# Serialize f. Use a string to save to file instead
io = PipeBuffer()
x_shape = (:W, :H, 2, :Batch)
y_shape = (4, :Batch)
onnx(io, f, x_shape, y_shape)

# Deserialize as a NaiveNASflux CompGraph (or use the deserialization method from ONNX.jl)
g = CompGraph(io)

x = ones(Float32, 5,4,2,3)
y = ones(Float32, 4, 3)
@test g(x,y) ≈ f(x,y)

# Serialization of CompGraphs does not require input shapes to be provided as they can be inferred.
io = PipeBuffer()
onnx(io, g)

g = CompGraph(io)
@test g(x,y) ≈ f(x,y)
```

## Supported Operations

```
Add
AveragePool
BatchNormalization
Concat
Conv
Dropout
Elu
Gemm
GlobalAveragePool
MaxPool
ReduceMean
Relu
Reshape
Selu
Squeeze
```

## Adding Operations

While the list of supported operations is currently quite meager, it is relatively straightforward to add support for more.

Serialization uses a lightweight graph tracing mechanism. More precisely, assuming one wants to map an ONNX operation to the function `myfun(args::SomeType....)`. What is needed is to implement a method `myfun(args::AbstractProbe...)` which
1. Adds ONNX information to one of the inputs (does not matter which one)
2. Returns at least one `AbstractProto` with information for the next function

This function typically looks something like this:

```julia
import ONNXmutable: AbstractProbe, recursename, nextname, newfrom, add!, name
function myfun(probes::AbstractProbe...)
    p = probes[1] # select any probe
    optype = "MyOpType"
    # Naming strategy (e.g. how to avoid duplicate names) is provided by the probe
    # Not strictly needed, but the onnx model is basically corrupt if duplicates exist
    nodename = recursename(optype, nextname(p))

    # Add ONNX node info
    add!(p, ONNX.Proto.NodeProto(
    # Names of input is provided by probes. This is why new probes need to be provided as output
    input = collect(name.(probes)),
    # Name of output from this node
    output = [nodename],
    op_type = optype))

    # Probes can procreate like this
    return newfrom(p, nodename)
end
```
See [serialize.jl](src/serialize/serialize.jl) for existing operations.


Deserialization is done by simply mapping operation types to functions in a dictionary in a very similar manner as it is done in [ONNX.jl](https://github.com/FluxML/ONNX.jl). This allows for both easy extension as well as overwriting of existing mappings with own implementations:

```julia
import ONNXmutable: actfuns

# All inputs which are not output from another node in the graph are provided in the method call
actfuns[:MyActFun1] = (params, α, β) -> x -> x^α + β
# Params contains a Dict with attributes.
actfuns[:MyActFun2] = function(params)
    α = get(params, :alpha, 1)
    return x -> α / x
end
ONNXmutable.refresh()
```
Note: After adding/changing an operation mapping one needs to call `ONNXmutable.refresh()` for it to take effect.
See [ops.jl](src/deserialize/ops.jl) for existing operations.


## Contributing

All contributions are welcome. Please file an issue before creating a PR.
