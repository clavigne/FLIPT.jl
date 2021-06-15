using JLD
using HDF5
using LinearAlgebra
using DifferentialEquations
include("NLSolve.jl")
include("GaussianField.jl")

ħ = 0.6582119514

function FLIPT_method(Norder, e0, mu, rho0, Ew, T, times, wmin, wmax)
    # make propagation quantities
    y0 = NLSolve.steady_state(rho0 , T)
    G! = NLSolve.make_G!(e0, ħ=ħ)

    function pt_increase(y, mu, cf)
        return G!(NLSolve.E(NLSolve.V!(NLSolve.ket_deepcopy(y), mu, ħ=ħ), cf))
    end

    field = NLSolve.Interaction([wmin], [wmax], [Ew], [false])
    cfield = NLSolve.compile_interaction(field + conj(field), y0)

    # compute orders
    # compile first:
    pt_increase(y0, mu, cfield)

    pt_out = [y0]
    timings = []
    for i in 1:Norder
        ti = time()
        push!(pt_out, pt_increase(pt_out[end], mu, cfield))
        to = time()
        push!(timings, to-ti)
    end

    # TD eval (in lisp haha)
    output = Array{Complex{Float64}}(undef, Norder+1, N,N, length(times))
    for j in 0:Norder
        for i in 1:length(times)
            @views NLSolve.at_t!(output[j+1,:,:,i], pt_out[j+1], times[i])
        end
    end
    return output, timings
end

function direct_to_pt(Amps, direct_out_raw)
    N,N,ntimes = size(direct_out_raw[1])
    Norder = length(direct_out_raw) - 1
    ys = transpose(hcat([d[:] for d in direct_out_raw]...))
    xs = vcat(0.0, Amps)
    X = hcat([xs.^n for n in 0:Norder]...)
    g = X' * X
    beta = g \ X' * ys


    output = Array{Complex{Float64}}(undef, Norder+1, N, N, ntimes)
    for n in 0:Norder
        output[n+1, :, :, :] = reshape(transpose(beta[n+1, :]), N,N,ntimes) 
    end

    return output
end

function direct_method(Norder, e0, mu, rho0, field, T, times; reltol=1e-8, abstol=1e-8)
    # compute the same thing but using a direct method
    Amps = [1.0/n for n in 1:Norder]
    pref = 1/(im * ħ)
    L = repeat(e0, 1, N)
    liouville =
        let L0 = pref .* (L .- L'), V = pref .* mu, field=Et
            ( rho, p, t) -> (L0 .* rho
                             + (p*field(t)) .* (V * rho - rho * V))
        end

    direct_out_raw = [repeat(reshape(rho0, N, N, 1), 1, 1, length(times))]

    # make sure everything is compiled first
    prob = ODEProblem(liouville, rho0, (-1.0, 1.0), 2.0)
    sol = solve(prob, DP5(),
                reltol=reltol,
                abstol=abstol
                )

    timings = []
    for A in Amps
        prob = ODEProblem(liouville, rho0, (-T, T), A)
        ti = time()
        sol = solve(prob, DP5(),
                    reltol=reltol,
                    abstol=abstol,
                    tstops=[0.0])
        to = time()
        push!(timings, to-ti)

        rhot_direct = Array{Complex{Float64}}(undef,N,N, length(times))
        for i in 1:length(times)
            @views rhot_direct[:,:,i] = sol(times[i]) 
        end
        push!(direct_out_raw, rhot_direct)
    end
    return direct_to_pt(Amps, direct_out_raw), timings
end


# standard run, short pulse
println("Comparisons (fig1)")
T = 300.0
w0 = 2.0/ħ
FWHM = 30.0
Norder = 4
times = collect(-FWHM:1.0:T)
A = 0.05

e0 = [0.0; 1.95; 2.05; 0.05]
N = length(e0)
mu = zeros(Complex{Float64}, N,N)
mu[1, 2] = 0.1
mu[1, 3] = 0.2
mu[4, 2] =-0.25
mu[4, 3] = 0.15
mu = mu + mu'
rho0 = diagm(0=>[1; 0; 0; 0]) .+ 0.0im

# make field
sigma = fwhm2sigma(FWHM)
Et, Ew = make_gaussian_field(A, w0, sigma)
wmin = w0-5*sigma
wmax = w0+5*sigma

result_flipt, timf = FLIPT_method(Norder, e0, mu, rho0, Ew, T,times, wmin, wmax)
result_direct, timd = direct_method(Norder+2, e0, mu, rho0, Et, T,times, abstol=1e-12)


println("flipt: ", sum(timf), " | direct: ", timd[end])
save("output/compare/short.jld", "times", times, "flipt", result_flipt, "direct", result_direct)


# standard run, long pulse
println("long pulse")
T = 3000.0
FWHM = 300.0
Norder = 4
times = collect(-FWHM:1.0:T)
A = 0.05

# make field
sigma = fwhm2sigma(FWHM)
Et, Ew = make_gaussian_field(A, w0, sigma)
wmin = w0-5*sigma
wmax = w0+5*sigma

result_flipt, timf = FLIPT_method(Norder, e0, mu, rho0, Ew, T,times, wmin, wmax)
result_direct, timd = direct_method(Norder, e0, mu, rho0, Et, T,times)
println("flipt: ", sum(timf), " | direct: ", timd[1])
save("output/compare/long.jld", "times", times, "flipt", result_flipt, "direct", result_direct)

Ntimes = 1 # paper results uses Ntimes=10

# standard run, timings scaling
println("Scaling (fig3) ")
scales = repeat(range(1.0, stop=10.0, length=20), Ntimes)
init_T = 3000.0
init_FWHM =300.0
times_flipt = []
times_direct = []
for k in scales
    T = k * init_T
    FWHM = k * init_FWHM
    Norder = 4
    times = collect(-FWHM:4.0 * k:T)

    # make field
    sigma = fwhm2sigma(FWHM)
    Et, Ew = make_gaussian_field(A, w0, sigma)
    wmin = w0-5*sigma
    wmax = w0+5*sigma

    result_flipt, timf = FLIPT_method(Norder, e0, mu, rho0, Ew, T,times, wmin, wmax)
    result_direct, timd = direct_method(Norder, e0, mu, rho0, Et, T,times)
    push!( times_direct, timd[1]) 
    push!( times_flipt, timf)

    println(k, " flipt: ", sum(timf), " | direct: ", timd[1]/k)
end
save("output/compare/scaled_timings.jld", "scales", scales, "FWHM", init_FWHM .* scales,
     "T", init_T .* scales, "flipt", times_flipt, "direct", times_direct)

# standard run, timings absolute
println("Scaling (fig2) ")
scales = repeat(range(1.0, stop=10.0, length=20), Ntimes)
times_flipt2 = []
npoints = []
for k in scales
    T = k * 3000.0
    FWHM = 300.0
    Norder = 4
    times = collect(-FWHM:4.0 * k:T)

    # make field
    sigma = fwhm2sigma(FWHM)
    Et, Ew = make_gaussian_field(A, w0, sigma)
    wmin = w0-5*sigma
    wmax = w0+5*sigma

    result_flipt, timf = FLIPT_method(Norder, e0, mu, rho0, Ew, T,times, wmin, wmax)
    push!( times_flipt2, timf)
    Δ = 2 * pi / T
    push!(  npoints, (wmax-wmin)/Δ )
    println(npoints[end], "  ", sum(timf))
end
save("output/compare/absolute_timings.jld", "flipt", times_flipt2, "npoints", npoints)


