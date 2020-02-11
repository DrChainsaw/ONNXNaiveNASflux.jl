
@testset "Reshape" begin
    using NaiveNASflux
    import ONNXmutable: Reshape

    dv(name, invertex, outsize) = mutable(name, Dense(nout(invertex), outsize), invertex)
    cv(name, invertex, outsize) = mutable(name, Conv((3,3), nout(invertex) => outsize, pad=(1,1)), invertex)
    rv(name, invertex, outsize, dims) = absorbvertex(Reshape(dims), outsize, invertex; traitdecoration=t -> NamedTrait(t, name))

    @testset "Reshape fixed nout" begin
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

    @testset "Reshape variable nout" begin

    end

end
