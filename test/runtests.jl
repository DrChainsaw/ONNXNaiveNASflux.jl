using ONNXNaiveNASflux, Flux
import ONNXNaiveNASflux: ONNX
using Test

@testset "ONNXNaiveNASflux.jl" begin

    @info "Test BaseOnnx"
    @testset "BaseOnnx" begin  
        include("baseonnx/readwrite.jl")
    end

    @info "Test Deserialization"
    @testset "Deserialize" begin
        include("deserialize/vertex.jl")
        include("deserialize/constraints.jl")
        include("deserialize/testdata.jl")
        include("deserialize/deserialize.jl")
    end

    @info "Test Serialization"
    @testset "Serialize" begin
        include("serialize/serialize.jl")
    end

    @info "Test validation"
    include("validate.jl")

    @info "Test README examples"
    include("examples.jl")
end
