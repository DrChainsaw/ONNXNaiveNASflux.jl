

"""
    validate(mp::ONNX.ModelProto)
    validate(mp::ONNX.ModelProto, fs...)

Validate `mp`, throwing an exception if it is invalid.

It is possible to specify the validation steps `fs` to perform. Default is `uniqueoutput, optypedefined, outputused, inputused, hasname`
"""
validate(mp::ONNX.ModelProto, fs...=(uniqueoutput, optypedefined, outputused, inputused, hasname)...) = foreach(f -> f(mp), fs)

errinfo(n::ONNX.NodeProto) = "NodeProto with: \n"  * join(["\t$pn:\t$(clstring(getproperty(n, pn)))" for pn in propertynames(n) if hasproperty(n, pn)], "\n")
clstring(x::AbstractArray) = "[" * join(string.(x), ", ") * "]"
clstring(x) = string(x)

"""
    uniqueoutput(mp::ONNX.ModelProto, or=error)
    uniqueoutput(gp::ONNX.GraphProto, or=error)

Test that output names are unique. If not, an error message will be passed to `or`.
"""
uniqueoutput(mp::ONNX.ModelProto, or=error) = uniqueoutput(mp.graph, or)
function uniqueoutput(gp::ONNX.GraphProto, or=error)
    d = Dict()
    for n in gp.node
        any_nonempty = false
        for oname in n.output
            # Empty names to signal positional outputs that are not used (e.g. generate output nr 2 but not nr 1)  
            # Therefore duplicates are allowed
            isempty(oname) && continue
            any_nonempty = true
            if haskey(d, oname)
                or("Duplicate output name: $oname found in \n$(errinfo(d[oname])) \nand\n $(errinfo(n))")
            end
            d[oname] = n
        end
        if !any_nonempty
            or("No selected output found for: \n$(errinfo(n))")
        end
    end
end

"""
    optypedefined(mp::ONNX.ModelProto, or=error)
    optypedefined(gp::ONNX.GraphProto, or=error)

Test that operations are defined for each node. If not, an error message will be passed to `or`.
"""
optypedefined(mp::ONNX.ModelProto, or=error) = optypedefined(mp.graph, or)
function optypedefined(gp::ONNX.GraphProto, or=error)
    for n in gp.node
        isempty(n.op_type) && or("No op_type defined for $(errinfo(n))")
    end
end

"""
    outputused(mp::ONNX.ModelProto, or=error)
    outputused(gp::ONNX.GraphProto, or=error)

Test that all outputs are used. If not, an error message will be passed to `or`.
"""
outputused(mp::ONNX.ModelProto, or=error) = outputused(mp.graph, or)
function outputused(gp::ONNX.GraphProto, or=error)
    found, used = ioused(gp)
    unusedouts = setdiff(found, used)
    str(s) = join(sort(collect(s)), ", ")
    isempty(unusedouts) || or("Found unused outputs: $(str(unusedouts))")
end

"""
    inputused(mp::ONNX.ModelProto, or=error)
    inputused(gp::ONNX.GraphProto, or=error)

Test that all inputs are used. If not, an error message will be passed to `or`.
"""
inputused(mp::ONNX.ModelProto, or=error) = inputused(mp.graph, or)
function inputused(gp::ONNX.GraphProto, or=error)
    used, found = ioused(gp)
    unusedins = setdiff(found, used)
    str(s) = join(sort(collect(s)), ", ")
    isempty(unusedins) || or("Found unused inputs: $(str(unusedins))")
end

function ioused(gp::ONNX.GraphProto)
    found = union(Set(name.(gp.input)), Set(name.(gp.initializer)))
    used = Set(name.(gp.output))
    for n in gp.node
        # Empty names to signal positional outputs that are not used (e.g. generate output nr 2 but not nr 1)  
        foreach(oname -> push!(found, oname), filter(!isempty, n.output))
        foreach(iname -> push!(used, iname), n.input)
    end

    return found, used
end

hasname(mp::ONNX.ModelProto, or=error) = hasname(mp.graph, or)
function hasname(gp::ONNX.GraphProto, or=error)
     isempty(gp.name) && or("Graph name is empty string!")
 end
