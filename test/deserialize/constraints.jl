
@testset "Reshape" begin
    using NaiveNASflux
    import ONNXmutable: Reshape, SizePseudoTransparent

    dv(name, invertex, outsize) = mutable(name, Dense(nout(invertex), outsize), invertex)
    cv(name, invertex, outsize) = mutable(name, Conv((3,3), nout(invertex) => outsize, pad=(1,1)), invertex)
    rv(name, invertex, outsize, dims) = absorbvertex(Reshape(dims), outsize, invertex; traitdecoration=t -> NamedTrait(SizePseudoTransparent(t), name))

    @testset "Reshape 2D -> 4D variable batch" begin
        v0 = inputvertex("in", 3)
        v1 = dv("v1", v0, 4)
        v2 = rv("v2", v1, 3, (5,2,3,Colon()))
        v3 = cv("v3", v2, 4)

        g = CompGraph(v0, v3)

        @test size(g(ones(3, 15))) == (5,2,4,2)

        @test_logs (:warn, r"Could not change nout") Δnout(v1, 3)
        apply_mutation(g)

        @test nout.(vertices(g)) == [3, 6, 3, 4]
        @test size(g(ones(3, 15))) == (5,2,4,3)
    end

    @testset "Reshape 2D -> 4D variable shape" begin
        v0 = inputvertex("in", 3)
        v1 = dv("v1", v0, 4)
        v2 = rv("v2", v1, 3, (5, Colon(),3, 2))
        v3 = cv("v3", v2, 4)

        g = CompGraph(v0, v3)

        @test size(g(ones(3, 15))) == (5,2,4,2)

        @test_logs (:warn, r"Could not change nout") Δnout(v1, 3)
        apply_mutation(g)

        @test nout.(vertices(g)) == [3, 6, 3, 4]
        @test size(g(ones(3, 15))) == (5,3,4,2)
    end

    @testset "Reshape 2D -> 4D variable nout" begin
        v0 = inputvertex("in", 3)
        v1 = dv("v1", v0, 4)
        v2 = rv("v2", v1, 2, (5, 3, Colon(), 2))
        v3 = cv("v3", v2, 4)

        g = CompGraph(v0, v3)

        @test size(g(ones(3, 15))) == (5,3,4,2)

        @test_logs (:warn, r"Could not change nout") Δnout(v1, 3)
        apply_mutation(g)

        @test nout.(vertices(g)) == [3, 6, 3, 4]
        @test size(g(ones(3, 15))) == (5,3,4,2)
    end

    @testset "Reshape 4D -> 2D variable batch" begin
        v0 = inputvertex("in", 3)
        v1 = cv("v1", v0, 4)
        v2 = rv("v2", v1, 5, (5 ,Colon()))
        v3 = dv("v3", v2, 4)

        g = CompGraph(v0, v3)

        @test size(g(ones(5,2,3,2))) == (4, 16)

        @test_logs (:warn, r"Could not change nout") Δnout(v1, 3)
        apply_mutation(g)

        @test nout.(vertices(g)) == [3, 5, 5, 4]
        @test size(g(ones(5,2,3,2))) == (4, 20)
    end

    @testset "Reshape 4D -> 2D variable nout" begin
        v0 = inputvertex("in", 3)
        v1 = cv("v1", v0, 4)
        v2 = rv("v2", v1, 5, (Colon(), 16))
        v3 = dv("v3", v2, 4)

        g = CompGraph(v0, v3)

        @test size(g(ones(5,2,3,2))) == (4, 16)

        @test_logs (:warn, r"Could not change nout") Δnout(v1, 3)
        apply_mutation(g)

        @test nout.(vertices(g)) == [3, 16, 20, 4]
        @test size(g(ones(5,2,3,2))) == (4, 16)
    end
end
