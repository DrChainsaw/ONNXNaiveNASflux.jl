

default_namestrat(f) = name_runningnr()
default_namestrat(g::CompGraph) = default_namestrat(vertices(g))
function default_namestrat(vs::AbstractVector{<:AbstractVertex})
    # Even if all vertices have unique names, we can't be certain that no vertex produces more than one node
    # Therefore, we must take a pass through name_runningnr for each op even after we have mapped the vertex to a nextname. This is the reason for the v -> f -> namegen(v, "") wierdness
    namegen = name_runningnr()
    all(isnamed, vs) && length(unique(name.(vs))) == length(name.(vs)) && return v -> f -> namegen(v, "")
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

    return function(f, init="_0")
        bname = namefun(f)
        candname = bname * init
        next = -1
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
