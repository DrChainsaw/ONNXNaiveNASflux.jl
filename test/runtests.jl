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
        include("serialize/protos.jl")
        include("serialize/serialize.jl")
    end

    @info "Test validation"
    include("validate.jl")

    @info "Test README examples"
    include("examples.jl")
end
