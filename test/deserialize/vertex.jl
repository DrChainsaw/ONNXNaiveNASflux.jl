@testset "Vertex" begin

    @testset "SourceVertex" begin
        using ONNXNaiveNASflux.NaiveNASflux
        import ONNXNaiveNASflux: SourceVertex

        data = randn(3,4,5)
        sv = SourceVertex(data, "sv")

        @test name(sv) == "sv"
        @test sv() == data

        @test nout(sv) == 3
        @test nin(sv) == []
    end
end
