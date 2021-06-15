module LVC
using LinearAlgebra
using SparseArrays

function vib(nvib)
    a = zeros(nvib,nvib)
    for k in 1:nvib-1
        a[k, k+1] = sqrt(k)
    end
    return a, deepcopy(a')
end

function S(i,j, n)
    out = zeros(n,n)
    out[i,j] = 1
    out[j,i] = 1
    return out
end

function to_sparse_elements(names, basis, H...)
    Hm = [Dict{String, Any}() for i in 1:length(H)]
    for m in names
        N = basis[m]
        if m == "el"
            @eval S(i,j) = S(i,j,$N)
        else
            a, ad = vib(N)
            @eval q = ($a + $ad)  .* sqrt(0.5)
            @eval p = ($a - $ad)  .* im *sqrt(0.5)
            @eval W = $ad * $a + 0.5 * I
        end

        for (k,el) in enumerate(H)
            expr = get(el, m, nothing)
            if expr != nothing
                if typeof(expr) == String
                    evaluated = sparse(eval(Meta.parse(expr)))
                    Hm[k][m] = evaluated
                else
                    Hm[k][m] = sparse(expr)
                end
            end
        end
    end 
    return Hm
end

function full_sparse(names, basis, Hm)
    if length(names)>1 
        out = sum([
            kron([
                get(el, m, sparse(I, basis[m], basis[m])) for m in names]...)
            for el in Hm])
        return out
    else
        m = names[1]
        return sum([get(el, m, sparse(I, basis[m], basis[m])) for el in Hm])
    end
end

function fkron(names, basis, op...)
    return full_sparse(names, basis, to_sparse_elements(names, basis, op...))
end

function electronic_projectors(names, basis, elindices...)
    out = (fkron(names, basis,
                    Dict("el" => let nel = basis["el"] 
                                     v = zeros(nel)
                                     v[k] = 1
                                     reshape(v, nel, 1)
                                  end
                         )) for k in elindices)
    if length(elindices) == 1
        return first(out)
    else
        return out
    end
end

end
