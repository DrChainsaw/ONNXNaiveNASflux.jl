using ONNXmutable
using Test


@testset "ONNXmutable.jl" begin

    @testset "Deserialize" begin
        include("deserialize/testdata.jl")
        include("deserialize/deserialize.jl")

    end


end
