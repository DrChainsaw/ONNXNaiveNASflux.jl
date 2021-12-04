
import ONNXRunTime
function onnxruntime_infer(f, inputs...)

	reversedims(a::AbstractArray{T,N}) where {T, N} = permutedims(a, N:-1:1)
	
	mktempdir() do dir
		modelfile = joinpath(dir, "model.onnx")
		save(modelfile, f, size.(inputs)...)

		model = ONNXRunTime.load_inference(modelfile)
		return model(Dict(ONNXRunTime.input_names(model) .=> reversedims.(inputs))) |> values .|> reversedims |> Tuple
	end

end
