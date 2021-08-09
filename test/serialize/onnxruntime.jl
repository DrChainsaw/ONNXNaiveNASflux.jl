
using PyCall
using Conda


function onnxruntime_infer(f, inputs...)
	if "onnxruntime" ∉ Conda._installed_packages()
		# TODO: Add some kind of warning if an incompatible default python installation is used due to ENV["PYTHON"] not being set to ""
		Conda.pip_interop(true)
		Conda.pip("install --no-warn-script-location", "onnxruntime==1.4")
	end

	modfile = "tmpmodel.onnx"
	try
		save(modfile, f, size.(inputs)...)

		ort = pyimport("onnxruntime")
		sess = ort.InferenceSession(modfile);
		ins = Dict(map(ii -> ii.name, sess.get_inputs()) .=> PyReverseDims.(inputs))
		return JuReverseDims.(Tuple(sess.run(nothing, ins)))
	finally
		rm(modfile; force=true)
	end
end

JuReverseDims(a::PyObject) = JuReverseDims(PyArray(a))
JuReverseDims(a::PyArray) = JuReverseDims(Array(a))
JuReverseDims(a::AbstractArray{T,N}) where {T, N} = permutedims(a, N:-1:1)
