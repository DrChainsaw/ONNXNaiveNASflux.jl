using ONNXmutable
using Test

@testset "ONNXmutable.jl" begin

    @info "Test Deserialization"
    @testset "Deserialize" begin
        include("deserialize/testdata.jl")
        include("deserialize/deserialize.jl")
    end

    @info "Test Serialization"
    @testset "Serialize" begin
        import ONNX
        include("serialize/tensorproto.jl")
        include("serialize/serialize.jl")
    end
end
