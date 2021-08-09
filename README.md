# ONNXNaiveNASflux

[![Build status](https://github.com/DrChainsaw/ONNXNaiveNASflux.jl/workflows/CI/badge.svg?branch=master)](https://github.com/DrChainsaw/ONNXNaiveNASflux.jl/actions)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/ONNXNaiveNASflux.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/ONNXNaiveNASflux-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/ONNXNaiveNASflux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/ONNXNaiveNASflux.jl)

[ONNX](https://onnx.ai) import and export for [Flux](https://github.com/FluxML/Flux.jl).

Models are imported as [NaiveNASflux](https://github.com/DrChainsaw/NaiveNASflux.jl) graphs, meaning that things like removing/inserting layers and pruning pre-trained models is a breeze.

Model export does not require the model to have any particular format. Almost any julia function can be exported as long as the primitives are recognized by ONNXNaiveNASflux. 

## Basic usage

```julia
Pkg.add(url="https://github.com/DrChainsaw/ONNXNaiveNASflux.jl")
```

Exporting is done using the `onnx` function which accepts a filename `String` or an `IO` as first argument:

```julia
# Save model as model.onnx where inputshapes are tuples with sizes of input.
save("model.onnx", model, inputshapes...)

# Load model as a CompGraph
graph = load("model.onnx", inputshapes...)
```
Input shapes can be omitted in which case an attempt to infer the shapes will be made. If supplied, one tuple with size as the dimensions of the corresponding input array (including batch dimension) is expected. 

Elements of input shape tuples can have one of the following types:
* `Integer`: The size of the corresponding dimension
* `Missing`: No shape info will be recorded for this dimension
* `Symbol` : Use the provided symbol as a variable name in the exported ONNX model

Names can be attached to inputs by providing a `Pair` where the first element is the name as a string, for example `"imageinput" => (:W, :H, 3, missing)`. Note that non-integer input sizes will be ignored when loading a model.

More elaborate example with a model defined as a plain Julia function:

```julia
using ONNXNaiveNASflux, Test, Statistics

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
save(io, f, x_shape, y_shape)

# Deserialize as a NaiveNASflux CompGraph
g = load(io)

x = ones(Float32, 5,4,2,3)
y = ones(Float32, 4, 3)
@test g(x,y) ≈ f(x,y)

# Serialization of CompGraphs does not require input shapes to be provided as they can be inferred.
io = PipeBuffer()
save(io, g)

g = load(io)
@test g(x,y) ≈ f(x,y)
```

## Supported Operations

```
Add
AveragePool
BatchNormalization
Concat
Constant
Conv
Dropout
Elu
Flatten
Gemm
GlobalAveragePool
GlobalMaxPool
LSTM
MaxPool
Mul
RNN
ReduceMean
Relu
Reshape
Selu
Softmax
Squeeze
Tanh
```

## Adding Operations

While the list of supported operations is currently quite meager, it is relatively straightforward to add support for more.

Serialization uses a lightweight graph tracing mechanism where `AbstractProbe`s are sent through the function to collect all ONNX operations they encounter.

To map the function `myfun(args::SomeType....)` to an ONNX operation one just defines a method `myfun(args::AbstractProbe...)` which
1. Adds ONNX information to one of the inputs (does not matter which one)
2. Returns at least one `AbstractProbe` with information for the next function

This function typically looks something like this:

```julia
import ONNXNaiveNASflux: AbstractProbe, recursename, nextname, newfrom, add!, name
function myfun(probes::AbstractProbe...)
    p = probes[1] # select any probe
    optype = "MyOpType"
    # Naming strategy (e.g. how to avoid duplicate names) is provided by the probe
    # Not strictly needed, but the onnx model is basically corrupt if duplicates exist
    nodename = recursename(optype, nextname(p))

    # Add ONNX node info
    add!(p, ONNX.NodeProto(
    # Names of input is provided by probes. This is why new probes need to be provided as output
    input = collect(name.(probes)),
    # Name of output from this node
    output = [nodename],
    op_type = optype))

    # Probes can procreate like this
    return newfrom(p, nodename, s -> s)
end
```
See [serialize.jl](src/serialize/serialize.jl) for existing operations.


Deserialization is done by simply mapping operation types to functions in a dictionary. This allows for both easy extension as well as overwriting of existing mappings with own implementations:

```julia
import ONNXNaiveNASflux: actfuns

# All inputs which are not output from another node in the graph are provided in the method call
actfuns[:SomeOp] = (params, α, β) -> x -> x^α + β
# Params contains a Dict with attributes.
actfuns[:AnotherOp] = function(params)
    α = get(params, :alpha, 1)
    return x -> α / x
end
ONNXNaiveNASflux.refresh()
```
Note: After adding/changing an operation mapping one needs to call `ONNXNaiveNASflux.refresh()` for it to take effect.
See [ops.jl](src/deserialize/ops.jl) for existing operations.


## Contributing

All contributions are welcome. Please file an issue before creating a PR.
