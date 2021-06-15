include("pyr2.jl")
include("LVC.jl")
using JLD
using LinearAlgebra

# Setup the modes 
# J. Chem. Phys. 123, 064313 (2005) 
fname = "pyr2.jld"
N = Dict( 
    (("el", 3),
    ("10a" , 15),
     ("6a" , 50),))


names = ["el",  "10a", "6a"]
Elim =  E2 + 0.8 
mu02 = 1.0
mu12 = 0.0
mu01 = sqrt(0.2) 

kb = 8.617330350e-5            # eV
beta = 1/(kb*300)

Z,e,R = let H =  H
    println("building H")
    H = LVC.fkron(names, N, H...)

    println("building electronic projectors")
    P0, P1, P2 = LVC.electronic_projectors(names, N, 1, 2 ,3 )
    P12 = hcat(P1,P2) 

    println("building H0 and H12")
    H0 = Hermitian(Array(P0' * H * P0))
    H12 = Hermitian(Array(P12' * H * P12))

    println("diagonalizing H12")
    ef12, pf12 = eigen(H12)
    which = ef12 .< Elim
    e12 = ef12[which]
    R12 = P12 * pf12[:, which]

    println("diagonalizing H0")
    ef0, pf0 = eigen(H0)

    Z = exp.(-beta .* ef0)
    Z /= sum(Z)
    which = ef0 .< 0.8

    e0 = ef0[which]
    R0 = P0 * pf0[:, which]
    Z0 = Z[which]/sum(Z[which])
    
    R = hcat(R0, R12)
    vcat(Z0, zeros(length(e12))), vcat(e0, e12) .- e0[1], R
end

function build_op(R, names, N, d)
    o = LVC.fkron(names, N, d)
    oR = o * R
    return R' * oR
end

mu = build_op(R, names, N,
              Dict("el" => "$mu02 * S(1,3) + $mu12 * S(2,3) + $mu01 * S(1,2)"))
p1 = build_op(R, names, N,
              Dict("el" => "S(2,2)"))
p2 = build_op(R, names, N,
              Dict("el" => "S(3,3)"))
wf0 = Array(R' * LVC.fkron(names, N, Dict("el"=>"S(1,3)"))[:,1])

println("number of states ", length(e))
println("saving...")

save(fname, "Z", Z, "e", e, "mu", mu,
     "P1", p1, "P2", p2, "wf0", wf0)


