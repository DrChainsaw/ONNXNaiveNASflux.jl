
using PyCall
using Conda


function onnxruntime_infer(f, inputs...)
	if "onnxruntime" ∉ Conda._installed_packages()
		pip = joinpath(Conda.SCRIPTDIR, "pip")
		run(`$pip install onnxruntime --no-warn-script-location`)
		ENV["PYTHON"]=Conda.PYTHONDIR
	end

	modfile = "tmpmodel.onnx"
	try
		# TODO: Alot of functions do not handle input->output shapes. Make them do so!!
		onnx(modfile, f, map(inpt -> ntuple(i -> missing, ndims(inpt)), inputs)...)

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
