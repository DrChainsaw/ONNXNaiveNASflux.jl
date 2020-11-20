@testset "Vertex" begin

    @testset "SourceVertex" begin
        using ONNXmutable.NaiveNASflux
        import ONNXmutable: SourceVertex

        data = randn(3,4,5)
        sv = SourceVertex(data, "sv")

        @test name(sv) == "sv"
        @test sv() == data

        @test nout(sv) == 3
        @test nin(sv) == []

        @test ismissing(minΔninfactor(sv))
        @test ismissing(minΔnoutfactor(sv))
    end

end
