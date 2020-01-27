

"""
    validate(mp::ONNX.Proto.ModelProto)
    validate(mp::ONNX.Proto.ModelProto, fs...)

Validate `mp`, throwing an exception if it is invalid.

It is possible to specify the validation steps `fs` to perform. Default is `uniqueoutput, optypedefined, outputused, inputused`
"""
validate(mp::ONNX.Proto.ModelProto, fs...=(uniqueoutput, optypedefined, outputused, inputused)...) = foreach(f -> f(mp), fs)

"""
    uniqueoutput(mp::ONNX.Proto.ModelProto, or=error)
    uniqueoutput(gp::ONNX.Proto.GraphProto, or=error)

Test that output names are unique. If not, an error message will be passed to `or`.
"""
uniqueoutput(mp::ONNX.Proto.ModelProto, or=error) = uniqueoutput(mp.graph, or)
function uniqueoutput(gp::ONNX.Proto.GraphProto, or=error)
    d = Dict()
    for n in gp.node
        for oname in n.output
            if haskey(d, oname)
                or("Duplicate output name: $oname found in \n $(d[oname]) \n and \n $n")
            end
            d[oname] = n
        end
    end
end

"""
    optypedefined(mp::ONNX.Proto.ModelProto, or=error)
    optypedefined(gp::ONNX.Proto.GraphProto, or=error)

Test that operations are defined for each node. If not, an error message will be passed to `or`.
"""
optypedefined(mp::ONNX.Proto.ModelProto, or=error) = optypedefined(mp.graph, or)
function optypedefined(gp::ONNX.Proto.GraphProto, or=error)
    for n in gp.node
        isdefined(n, :op_type) || or("No op_type defined for $n")
    end
end

"""
    outputused(mp::ONNX.Proto.ModelProto, or=error)
    outputused(gp::ONNX.Proto.GraphProto, or=error)

Test that all outputs are used. If not, an error message will be passed to `or`.
"""
outputused(mp::ONNX.Proto.ModelProto, or=error) = outputused(mp.graph, or)
function outputused(gp::ONNX.Proto.GraphProto, or=error)
    found, used = ioused(gp)
    unusedouts = setdiff(found, used)
    str(s) = join(sort(collect(s)), ", ")
    isempty(unusedouts) || or("Found unused outputs: $(str(unusedouts))")
end

"""
    inputused(mp::ONNX.Proto.ModelProto, or=error)
    inputused(gp::ONNX.Proto.GraphProto, or=error)

Test that all inputs are used. If not, an error message will be passed to `or`.
"""
inputused(mp::ONNX.Proto.ModelProto, or=error) = inputused(mp.graph, or)
function inputused(gp::ONNX.Proto.GraphProto, or=error)
    used, found = ioused(gp)
    unusedins = setdiff(found, used)
    str(s) = join(sort(collect(s)), ", ")
    isempty(unusedins) || or("Found unused inputs: $(str(unusedins))")
end

function ioused(gp::ONNX.Proto.GraphProto)
    found = Set(name.(gp.input))
    used = Set(name.(gp.output))
    for n in gp.node
        foreach(oname -> push!(found, oname), n.output)
        foreach(iname -> push!(used, iname), n.input)
    end

    return found, used
end
