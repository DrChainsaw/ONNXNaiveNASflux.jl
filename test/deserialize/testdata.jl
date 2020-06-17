using Pkg.Artifacts
import ONNX: get_array, readproto, Proto.TensorProto
import ONNXmutable:  CompGraphBuilder, extract

const last_dl_time = Dict()

function prepare_node_test(name, ninputs, noutputs)
    ahash = get_node_artifact(name, ninputs=ninputs, noutputs=noutputs)
    apath = artifact_path(ahash)
    model, sizes = extract(joinpath(apath, "model.onnx"))

    inputs = readinput.(apath, 0:ninputs-1)
    outputs = readoutput.(apath, 0:noutputs-1)

    graph = model.graph
    for (vi, val) in zip(graph.input, inputs)
        graph.initializer[vi.name] = val
    end

    gb = CompGraphBuilder(graph, sizes)
    return model, sizes, gb, inputs, outputs
end

# Browsable link: https://github.com/onnx/onnx/tree/v1.6.0/onnx/backend/test/data/node
function get_node_artifact(name, node=name; ninputs, noutputs, location = "https://raw.githubusercontent.com/onnx/onnx/v1.6.0/onnx/backend/test/data/node/")
    return get_artifact(name, joinpath(location, node), ninputs, noutputs)
end

# Copy paste from documentation basically
function get_artifact(name, location, ninputs, noutputs, rcnt=5)
    artifacts_toml = joinpath(@__DIR__, "Artifacts.toml")
    ahash = artifact_hash(name, artifacts_toml)
    if isnothing(ahash) || !artifact_exists(ahash)

        throttle_download(location)
        try
            ahash = create_artifact() do artifact_dir
                mkpath(joinpath(artifact_dir, "test_data_set_0"))

                download(location * "/model.onnx", joinpath(artifact_dir, "model.onnx"))
                for i in 0:ninputs-1
                    download(inputfile(location, i), inputfile(artifact_dir, i))
                end

                for i in 0:noutputs-1
                    download(outputfile(location, i), outputfile(artifact_dir, i))
                end
            end
        catch e
            rcnt == 0 && rethrow(e)
            @warn sprint(showerror, e) * " Retry nr $rcnt"
            get_artifact(name, location, ninputs, noutputs, rcnt-1)
        end

        bind_artifact!(artifacts_toml, name, ahash; force=true)
    end
    return ahash
end
    # Limit minimum time between downloads so the server doesn't block us
function throttle_download(location, waittime=0.5)
    ctime = time()
    baseurl = splitpath(location)[2]
    ptime = get(last_dl_time, baseurl, ctime-waittime)
    sleep(max(0, ptime-ctime + waittime))
    last_dl_time[baseurl] = time()
end


inputfile(apath, i) = apath * "/test_data_set_0" * "/" * join(["input_", i, ".pb"])
outputfile(apath, i) = apath * "/test_data_set_0" * "/" * join(["output_", i, ".pb"])

readinput(apath, i) = readdata(inputfile(apath, i))
readoutput(apath, i) = readdata(outputfile(apath, i))
readdata(filename) = readproto(open(filename), TensorProto()) |> get_array
