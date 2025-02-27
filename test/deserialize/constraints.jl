
@testset "Constraints" begin
    using ONNXNaiveNASflux.NaiveNASflux
    using ONNXNaiveNASflux: SizePseudoTransparent, MeasureNout
    using NaiveNASflux: actdim

    dv(name, invertex, outsize) = dv(name, invertex, nout(invertex), outsize) 
    dv(name, invertex, insize, outsize) = fluxvertex(name, Dense(insize, outsize), invertex)
    cv(name, invertex, outsize) = cv(name, invertex, nout(invertex), outsize) 
    cv(name, invertex, insize, outsize) =fluxvertex(name, Conv((3,3), insize => outsize, pad=(1,1)), invertex)

    @testset "Reshape" begin
        using ONNXNaiveNASflux: Reshape    

        rv(name, invertex, dims) = absorbvertex(name, MeasureNout(Reshape(dims)), invertex; traitdecoration=SizePseudoTransparent)

        @testset "Reshape 2D -> 4D variable batch" begin
            v0 = inputvertex("in", 3)
            v1 = dv("v1", v0, 4)
            v2 = rv("v2", v1, (5,2,3,Colon()))
            v3 = cv("v3", v2, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [3]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 3, 15))) == (5,2,4,2)

            @test_logs (:warn, r"Could not change nout") Δnout!(Returns(1), v1, 3)

            @test nout.(vertices(g)) == [3, 6, 3, 4]
            @test size(g(ones(Float32, 3, 15))) == (5,2,4,3)
        end

        @testset "Reshape 2D -> 4D variable shape" begin
            v0 = inputvertex("in", 3)
            v1 = dv("v1", v0, 4)
            v2 = rv("v2", v1, (5, Colon(),3, 2))
            v3 = cv("v3", v2, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [3]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 3, 15))) == (5,2,4,2)

            @test_logs (:warn, r"Could not change nout") Δnout!(Returns(1), v1, 3)
 
            @test nout.(vertices(g)) == [3, 6, 3, 4]
            @test size(g(ones(Float32, 3, 15))) == (5,3,4,2)
        end

        @testset "Reshape 2D -> 4D variable nout" begin
            v0 = inputvertex("in", 3)
            v1 = dv("v1", v0, 4)
            v2 = rv("v2", v1, (5, 3, Colon(), 2))
            v3 = cv("v3", v2, 2, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [3]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 3, 15))) == (5,3,4,2)

            # note that this works only because MeasureNout got the size when we evaluated the graph absorbvertex
            # In other cases one might need try_infer_sizes!
            @test_logs (:warn, r"Could not change nout") Δnout!(Returns(1), v1, 3)

            @test nout.(vertices(g)) == [3, 6, 3, 4]
            @test size(g(ones(Float32, 3, 15))) == (5,3,4,2)
        end

        @testset "Reshape 2D -> 4D variable nout 0-dim" begin
            v0 = inputvertex("in", 3)
            v1 = dv("v1", v0, 20)
            v2 = rv("v2", v1, (2, 5, Colon(), 0))
            v3 = cv("v3", v2, 2, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [3]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 3, 2))) == (2,5,4,2)
          
            # note that this works only because MeasureNout got the size when we evaluated the graph absorbvertex
            # In other cases one might need try_infer_sizes!
            @test_logs (:warn, r"Could not change nout") Δnout!(Returns(1), v1, 3)

            @test nout.(vertices(g)) == [3, 30, 3, 4]
            @test size(g(ones(Float32, 3, 2))) == (2,5,4,2)
        end

        @testset "Reshape 2D -> 4D all fixed" begin
            v0 = inputvertex("in", 3)
            v1 = dv("v1", v0, 4)
            v2 = rv("v2", v1, (5, 2, 3, 2))
            v3 = cv("v3", v2, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [3]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 3, 15))) == (5,2,4,2)

            @test @test_logs((:warn, r"Could not change nout"), match_mode=:any, Δnout!(Returns(1), v1, 3)) == false

            @test nout.(vertices(g)) == [3, 4, 3, 4]
            @test size(g(ones(Float32, 3, 15))) == (5,2,4,2)
        end

        @testset "Reshape 2D -> 4D all fixed 0-dim" begin
            v0 = inputvertex("in", 3)
            v1 = dv("v1", v0, 4)
            v2 = rv("v2", v1, (5, 2, 0, 2))
            v3 = cv("v3", v2, 4, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [3]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 3, 20))) == (5,2,4,2)

            @test_logs (:warn, r"Could not change nout") Δnout!(Returns(1), v1, 3)

            @test nout.(vertices(g)) == [3, 6, 6, 4]
            @test size(g(ones(Float32, 3, 20))) == (5,2,4,2)
        end

        @testset "Reshape 4D -> 2D variable batch" begin
            v0 = inputvertex("in", 3)
            v1 = cv("v1", v0, 4)
            v2 = rv("v2", v1, (5 ,Colon()))
            v3 = dv("v3", v2, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [1]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 16)

            @test_logs (:warn, r"Could not change nout") Δnout!(Returns(1), v1, 3)

            @test nout.(vertices(g)) == [3, 8, 10, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 16)
        end

        @testset "Reshape 4D -> 2D variable nout" begin
            v0 = inputvertex("in", 3)
            v1 = cv("v1", v0, 4)
            v2 = rv("v2", v1, (Colon(), 16))
            v3 = dv("v3", v2, 5, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [1]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 16)

            @test_logs (:warn, r"Could not change nout") Δnout!(Returns(1), v1, 3)

            @test nout.(vertices(g)) == [3, 16, 20, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 16)
        end

        @testset "Reshape 4D -> 2D variable nout 0-dim" begin
            v0 = inputvertex("in", 3)
            v1 = cv("v1", v0, 4)
            v2 = rv("v2", v1, (Colon(), 0))
            v3 = dv("v3", v2, 40, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [1]
            @test layer(v2) isa Reshape

            g = CompGraph(v0, v3)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 2)

            Δnout!(v1, -2)

            @test nout.(vertices(g)) == [3, 2, 20, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 2)
        end
    end


    @testset "Flatten" begin
        import ONNXNaiveNASflux: Flatten
        fv(name, invertex, dim) = absorbvertex(name, MeasureNout(Flatten(dim)), invertex; traitdecoration=SizePseudoTransparent)

        function tg(outsize, dim)
            v0 = inputvertex("in", 3)
            v1 = cv("v1", v0, 4)
            v2 = fv("v2", v1, dim)
            v3 = dv("v3", v2, outsize, 4)

            @test actdim(v2) == [actdim(layer(v2))] == [1]
            @test layer(v2) isa Flatten

            g = CompGraph(v0, v3)
        end

        @testset "Flatten 4D dim 0" begin
            g = tg(80, 4)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 1)

            Δnout!(vertices(g)[2], -3)

            @test nout.(vertices(g)) == [3, 1, 20, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 1)
        end

        @testset "Flatten 4D dim 1" begin
            g = tg(5, 1)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 16)

            Δnout!(vertices(g)[2], 3)

            @test nout.(vertices(g)) == [3, 7, 5, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 28)
        end

        @testset "Flatten 4D dim 2" begin
            g = tg(10, 2)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 8)

            Δnout!(vertices(g)[2], -1)

            @test nout.(vertices(g)) == [3, 3, 10, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 6)
        end

        @testset "Flatten 4D dim 3" begin
            g = tg(40, 3)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 2)

            Δnout!(vertices(g)[2], -2)

            @test nout.(vertices(g)) == [3, 2, 20, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 2)
        end

        @testset "Flatten 4D dim 4" begin
            g = tg(80, 4)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 1)

            Δnout!(vertices(g)[2], -3)

            @test nout.(vertices(g)) == [3, 1, 20, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 1)
        end

        @testset "Flatten 4D dim -4" begin
            g = tg(1, -4)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 80)

            Δnout!(vertices(g)[2], -2)

            @test nout.(vertices(g)) == [3, 2, 1, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 40)
        end

        @testset "Flatten 4D dim -3" begin
            g = tg(5, -3)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 16)

            Δnout!(vertices(g)[2], 3)

            @test nout.(vertices(g)) == [3, 7, 5, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 28)
        end

        @testset "Flatten 4D dim -2" begin
            g = tg(10, -2)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 8)

            Δnout!(vertices(g)[2], -1)

            @test nout.(vertices(g)) == [3, 3, 10, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 6)
        end

        @testset "Flatten 4D dim -1" begin
            g = tg(40, -1)

            @test size(g(ones(Float32, 5,2,3,2))) == (4, 2)

            Δnout!(vertices(g)[2], -2)

            @test nout.(vertices(g)) == [3, 2, 20, 4]
            @test size(g(ones(Float32, 5,2,3,2))) == (4, 2)
        end
    end
end
