module NLSolve
using LinearAlgebra
using IterativeSolvers
import Base: *

mutable struct Ket{T<:AbstractFloat}
    order::Int64
    dω::T
    η::T
    y::Array{Complex{T},3}      # the solution array y
    shifts::Array{Int64,1}      # the w shifts of each group
    indptrs::Array{Int64,1}      # the starting element of each group
end

struct Interaction
    wmin
    wmax
    E
    reversed
end

mutable struct CompiledInteraction{T<:AbstractFloat}
    lls::Array{Int64,1}
    uls::Array{Int64,1}
    E::Array{Complex{T}, 2}
    Eη::Array{Complex{T}, 2}
    η::T
end

####################################################################
function ket_deepcopy(ket::Ket{T}) where {T<:Any}
    return Ket(ket.order, ket.dω, ket.η, deepcopy(ket.y),
               deepcopy(ket.shifts), deepcopy(ket.indptrs))
end

function ket_freq_prune(y::Ket{T}, f) where {T<:Any}
    z = sparse2freqs(y)
    mask = [f(imag(zi)) for zi in z]
    return Ket(y.order,
               y.dω, y.η,
               y.y[:,mask,:], inds2sparse(sparse2inds(y)[mask])...)
end

function ket_state_prune(y::Ket{T}, f) where {T<:Any}
    mask = [f(y.y[:,i,:]) for i in 1:size(y.y)[2]]
    return Ket(y.order,
               y.dω, y.η,
               y.y[:,mask, :], inds2sparse(sparse2inds(y)[mask])...)
end

function ket_delay(y::Ket{T}, tau) where {T<:Any}
    new = ket_deepcopy(y)
    z = sparse2freqs(y)
    for k in 1:length(z)
        new.y[:, k, :] *= exp(z[k] .* tau)
    end
    return new 
end

####################################################################
function Base.:conj(a::Interaction)
    Interaction(a.wmin, a.wmax, a.E, broadcast(!, a.reversed))
end

function Base.:+(a::Interaction, b::Interaction)
    Interaction(
        vcat(a.wmin, b.wmin),
        vcat(a.wmax, b.wmax),
        vcat(a.E, b.E),
        vcat(a.reversed, b.reversed))
end

function compile_interaction(interaction, ket::Ket{T}) where {T<:Any}
    dω = ket.dω
    η = dω
    
    lls = Int64.(floor.(interaction.wmin/dω))
    uls = Int64.(ceil.(interaction.wmax/dω))
    ns = uls .- lls .+ 1
    
    E = zeros(Complex{T}, maximum(ns), length(lls))
    Eη = zeros(Complex{T}, maximum(ns), length(lls))

    for k in 1:length(lls)
        # Gridded field with and without eta 
        tf = map(
            wi -> interaction.E[k](
                wi .* dω) .* (dω/(2*pi)),
            collect(lls[k]:uls[k]))

        tf_with_eta = map(
            wi -> interaction.E[k](
                wi .* dω .- im * η) .* (dω/(2*pi)),
            collect(lls[k]:uls[k]))
        
        if interaction.reversed[k]
            tf = reverse(conj(tf))
            tf_with_eta = reverse(conj(tf_with_eta))

            ul = uls[k]
            ll = lls[k]
            uls[k] = -ll
            lls[k] = -ul
        end
        E[1:length(tf), k] .= tf
        Eη[1:length(tf), k] .= tf_with_eta
    end
    return CompiledInteraction(lls, uls, E, Eη, η)
end


####################################################################
function steady_state(y0, maxT; Ω₀=0.0im)
    # parameters
    dω = 2 * pi/maxT
    n = size(y0)[1]
    
    # make the initial state
    ket0 = Ket(0, dω, 0.0, Complex.(reshape(y0, n, 1, n)),
               [0], [1; 2])
    return ket0 
end

function sparse2freqs(ket::Ket{T}) where {T<:Any}
    z = zeros(Complex{T}, ket.indptrs[end]-1)
    wi = sparse2inds(ket)
    z[:] .= wi .* (ket.dω * im) .+ (ket.η)
    return z
end

function sparse2inds(ket::Ket)
    out = zeros(Int64, ket.indptrs[end]-1)
    
    for k in 1:length(ket.shifts)
        ind = ket.shifts[k]
        for i in ket.indptrs[k]:ket.indptrs[k+1]-1
            out[i] = ind
            ind += 1
        end
    end
    return out
end

function sparse2llul(ket::Ket)
    N = length(ket.shifts)
    lls = zeros(Int64, N)
    uls = zeros(Int64, N)

    for k in 1:N
        lls[k] = ket.shifts[k]
        uls[k] = ket.shifts[k] + ket.indptrs[k+1] - ket.indptrs[k] -1
    end
    return lls, uls
end

function inds2sparse(inds)
    # Set up sparse structure of output
    shifts = [inds[1]]
    indptrs = [1]
    
    last = inds[1]
    for kout in 1:length(inds)
        if inds[kout] - last > 1
            append!(shifts, inds[kout])
            append!(indptrs, kout)
        end
        
        last = inds[kout]
    end
    
    append!(indptrs, length(inds)+1)
    return shifts, indptrs
end

function new_indices(ket::Ket, interaction::CompiledInteraction)
    ll1, ul1 = sparse2llul(ket)
    return new_indices(ll1,ul1,interaction.lls,interaction.uls)
end

function new_indices(ket1::Ket, ket2::Ket)
    ll1, ul1 = sparse2llul(ket1)
    ll2, ul2 = sparse2llul(ket2)
    return new_indices(ll1,ul1,ll2,ul2)
end

function new_indices(
    lls1::Array{Int64,1}, uls1::Array{Int64,1},
    lls2::Array{Int64,1}, uls2::Array{Int64,1})
    # Create new indices from to sets of lower upper indices
    ninds = BitSet([])
    for k2 in 1:length(lls2)
        for k1 in 1:length(lls1)
            min = lls2[k2] + lls1[k1]
            max = uls2[k2] + uls1[k1]

            # many many allocations...
            # push!(ninds, collect(min:max)...)

            # much better 
            for kk in min:max
                push!(ninds, kk)
            end
        end
    end

    wout = sort!(collect(ninds))
    return wout
end


@inline function V_blascall(scal::T, op::Array{T,2}, y::Array{T,3}) where {T}
    # return scal * (op *y - y * op)
    n,m,n = size(y)
    out = similar(y)
    ONE = one(T)
    ZERO = zero(T)

    r_n_nm = reshape(view(y, :,:,:), n, n*m)
    out_n_nm = reshape(view(out, :,:,:), n, n*m)
    BLAS.gemm!('N', 'N', ONE, op, r_n_nm, ZERO, out_n_nm)
    
    r_nm_n = reshape(view(y, :,:,:),  n*m, n)
    out_nm_n = reshape(view(out, :,:,:),  n*m, n)
    BLAS.gemm!('N', 'N', -scal, r_nm_n, op, scal, out_nm_n)
    return out
end


@inline function E_blascall(E::Array{T,2}, yin::Array{T,3}) where {T}
    m1, m2 = size(E)
    n = size(yin)[1]
    yout = zeros(T, n, m2, n)
    ZERO =zero(T)
    ONE = one(T)
    
    vo = view(yout, 1:n, 1:m2, 1:n) 
    vi = view(yin, 1:n, 1:m1, 1:n) 
    for k in 1:n
        vo = view(yout, 1:n, 1:m2, k)
        vi = view(yin, 1:n, 1:m1, k)
        BLAS.gemm!('N', 'N', ONE, vi, E, ZERO, vo) 
    end
    return yout
end

function V!(inket::Ket{T},
            mu::Array{Complex{T},2}; ħ=1.0) where {T<:Any}
    inket.y = V_blascall(1.0/(im * ħ), mu, inket.y)
    return inket
end

function E(inket::Ket{T},
           interaction::CompiledInteraction{T};
           with_η=nothing,
           filter=nothing) where {T<:Any}

    lls = interaction.lls
    uls = interaction.uls

    if with_η == nothing
        # use η if the state has η == 0, to avoid breakage in G
        if inket.η == 0.0
            E = interaction.Eη
            η = interaction.η 
        else
            E = interaction.E
            η = 0.0
        end
    else
        if with_η
            E = interaction.Eη
            η = interaction.η 
        else
            E = interaction.E
            η = 0.0
        end
    end

    wout = new_indices(inket, interaction)
    if filter != nothing
        which = [filter(wi * inket.dω) for wi in wout]
        wout = wout[which]
    end

    win = sparse2inds(inket)
    
    nfields = length(lls)
    nin = length(win)
    nout = length(wout)
    n = size(inket.y)[1]

    emat = zeros(Complex{T}, nin, nout)
    ZERO = zero(Complex{T})

    for ei in 1:nfields
        # build Emat
        ll = lls[ei]
        ul = uls[ei]

        for kin in 1:nin
            for kout in 1:nout
                wi = wout[kout] - win[kin]
                if wi >= ll
                    if wi > ul
                        break
                    else
                        emat[kin, kout] += E[wi - ll + 1, ei] 
                    end
                end
            end
        end
    end

    # make the outgoing ket
    return Ket(inket.order+1, inket.dω, inket.η + η,
                E_blascall(emat, inket.y), inds2sparse(wout)...)

end

function make_G!(e0::Array{T,1}; deco=0.0, ħ=1.0) where {T<:Any}
    # Density matrix picture in eigenbasis
    # deco is real matrix of decoherence rates

    # Make the Green's function
    n::Int = length(e0)
    pref = 1/(im * ħ)
    L = repeat(e0, 1, n)
    L0::Array{Complex{T},2} = pref .* (L .- L' .+ deco .* im)


    function G!(inket::Ket{T})
        w = sparse2freqs(inket)
        m = length(w)
        Threads.@threads for j in 1:n
            for k in 1:m
                for i in 1:n
                    @inbounds inket.y[i,k,j] /= (w[k] - L0[i, j])
                end
            end
        end
        return inket
    end
    return G!
end

function make_G!(e0::Array{T,1}, deco::Array{T,2}, popmat::AbstractArray{T, 2}; ħ=1.0) where {T<:Any}
    # Make the Green's function
    n::Int = length(e0)
    pref = 1/(im * ħ)
    L = repeat(e0, 1, n)
    L0::Array{Complex{T},2} = pref .* (L .- L' .+ deco .* im) 
    Tmat::Array{Complex{T}, 2} = pref .* popmat .* im


    function G!(inket::Ket{T})
        w = sparse2freqs(inket)
        m = length(w)
        workspace = Array{Complex{T}}(undef, n, m)
        workspace1 = zeros(Complex{T},n)
        Threads.@threads for j in 1:n
            for k in 1:m
                # i<j
                for i in 1:j-1
                    inket.y[i,k,j] /= (w[k] - L0[i, j])
                end

                # diagonal elements i == j
                workspace[j,k] = inket.y[j,k,j]
                
                # i>j
                for i in j+1:n
                    inket.y[i,k,j] /= (w[k] - L0[i, j])
                end
            end
        end

        # with gmres
        initially_zero = true
        diaginds = diagind(Tmat)
        for k in 1:m
            # IterativeSolvers.gmres!(workspace1, w[k]*I - Tmat, view(workspace,:,k))
            workspace1 = (w[k] * I - Tmat) \ view(workspace, :, k)

            for j in 1:n
                inket.y[j, k, j] = workspace1[j] 
            end

            initially_zero = 1
        end



        return inket
    end
    return G!
end




###############################################################################
## Evaluators for time-dependent quantities 
function at_t!(out, y::Ket{T}, t) where {T<:Any}
    # Evaluate y at time t, overwriting out.

    # This function basically consists of sparse2inds with a time
    # evaluation.
    N = size(out)[1]

    if y.order == 0
        for k in 1:size(y.y)[2]
            for n in 1:N
                for m in 1:N
                    out[n,m] = sum(y.y[n, k, m])
                end
            end
        end
        return out
    end

    out .= zero(Complex{T})
    
    for k in 1:length(y.shifts)
        ind = y.shifts[k]
        for i in y.indptrs[k]:y.indptrs[k+1]-1
            expz = exp((ind * y.dω * im + y.η) * t)
            for n in 1:N
                for m in 1:N
                    out[n,m] += y.y[n,i,m] * expz
                end
            end
            ind += 1
        end
    end
    return out
end

function at_t(y::Ket{T}, t) where {T<:Any}
    N = size(y.y)[1]
    out = Array{Complex{T}}(undef, N,N)
    return at_t!(out, y, t)
end

function expect(O::Array{Complex{T}, 1},
                rho::Ket{T},
                t) where {T<:Any}
    nt = length(t)
    N = size(rho.y)[1]
    NN = N * N
    out = Array{Complex{T}}(undef, nt)
    tmp = Array{Complex{T}}(undef, N, N)
    for i in 1:nt
        at_t!(tmp, rho, t[i])
        out[i] = 0.0
        for n in 1:NN
            out[i] += conj(O[n]) * tmp[n]
        end
    end
    return out
end

function expect(O::Array{Complex{T},2},
                rho::Ket{T},
                t) where {T<:Any}
    n = size(O)[1]
    Oflat = reshape(copy(conj(O)), n*n) 
    return expect(Oflat, rho, t)
end

###############################################################################
## Evaluators for frequency-dependent quantities (Density matrix )
function take_spectrum(y::Ket{T}, interaction, mu) where {T<:Any} 
    out = []
    field = []
    w = []
    dw = y.dω
    η = y.η
    n = size(y.y)[1]

    lls = Int64.(floor.(interaction.wmin/dw))
    uls = Int64.(ceil.(interaction.wmax/dw))
    ns = uls .- lls .+ 1
    ws = sparse2inds(y)
    nws = length(ws)

    Z = zero(Complex{T})
    muT = copy(reshape(transpose(mu), n*n))
    tmp = Array{Complex{T}}(undef, n*n)
    
    for k in 1:length(lls)
        wi = collect(lls[k]:uls[k])
        if interaction.reversed[k] 
            x = reverse(-wi)
        else
            x = wi
        end

        mu_i = zeros(Complex{T}, length(wi)) 
        nwi = length(wi)

        for jw in 1:nws
            which = wi .== ws[jw]
            if any(which)
                tmp[:] = y.y[:,jw, :]
                val = muT ⋅ tmp
                mu_i[which] .+= val
            end
        end


        # compute the field
        E = zeros(Complex{T}, length(wi))
        for i in 1:length(wi)
            E[i] = interaction.E[k](
                wi[i] * dw + im * η)  
        end

        if interaction.reversed[k]
            E = reverse(conj(E))
        end

        w = [w, x .* dw]
        out = [out;   -2.0 .* mu_i .* conj(E) ]
    end
    return vcat(w...), vcat(out...)
end


pow_m1_n =
    let v = (-1, 1)
        n -> @inbounds v[mod1(n, 2)]
    end

function make_kernel(η::Real, T::Real)
    kernel(n) = 1/(2 * pi) * pow_m1_n(n) *
        (exp(-η * T/2) - exp( η * T/2)) / (2 * pi * im * n + T * η)
    return kernel
end

function take_signal(y::Ket{T}, mu) where {T<:Any} 
    wi = sparse2inds(y)
    mu_i = zeros(Complex{T}, length(wi)) 
    out = zeros(Complex{T}, length(wi)) 
    nwi = length(wi)
    muT = transpose(mu)
    ker = NLSolve.make_kernel(y.η, 2*pi/y.dω)

    for k in 1:nwi
        mu_i[k] = sum(muT .* y.y[:,k,:])
    end

    Z = zero(eltype(out))
    for i in 1:nwi
        accum = Z
        for j in 1:nwi
            accum += ker(wi[j] - wi[i]) * mu_i[j]
        end
        out[i] = accum
    end

    return wi .* y.dω, -out
end

end
