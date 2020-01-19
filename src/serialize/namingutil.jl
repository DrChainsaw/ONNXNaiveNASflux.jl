

default_namestrat(g::CompGraph) = default_namestrat(vertices(g))
function default_namestrat(vs::AbstractVector{<:AbstractVertex})
    all(isnamed, vs) && return v -> name(v)
    namegen = name_runningnr()
    ng(v::AbstractVertex) = namegen
    ng(f) = namegen(f)
    return ng
end

isnamed(v::AbstractVertex) = isnamed(base(v))
isnamed(v::CompVertex) = false
isnamed(v::InputVertex) = true
isnamed(v::MutationVertex) = isnamed(trait(v))
isnamed(t::DecoratingTrait) = isnamed(base(t))
isnamed(t) = false
isnamed(::NamedTrait) = true

function name_runningnr(namefun = genname)
    exists = Set{String}()

    return function(f)
        bname = genname(f)
        next = 0
        candname = bname * "_" * string(next)
        while candname in exists
            next += 1
            candname = bname * "_" * string(next)
        end
        push!(exists, candname)
        return candname
    end
end

genname(v::AbstractVertex) = name(v)
genname(f::F) where F = lowercase(string(F.name))
genname(s::AbstractString) = s
genname(f::Function) = lowercase(string(f))

recursename(f, namestrat) = recursename(f, namestrat(f))
recursename(f, fname::String) = fname
