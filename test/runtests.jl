using ONNXmutable
using Test

@testset "ONNXmutable.jl" begin

    @testset "Deserialize" begin
        include("deserialize/testdata.jl")
        include("deserialize/deserialize.jl")
    end

    @testset "Serialize" begin
        import ONNX
        include("serialize/tensorproto.jl")
        include("serialize/serialize.jl")
    end


end
