using JLD
using LinearAlgebra
using FFTW
include("NLSolve.jl")
include("GaussianField.jl")

ħ = 0.6582119514
ev_nm = 1239.842
ev_wn = 8065.54
kb = 8.617330350e-5            # eV

function spec2d(w, delays, TA)
    spec2d = fftshift(fft(TA,2),2)
    dw2 = 1/(delays[2]-delays[1])
    x2d = collect(range(-1.0, stop=1.0, length=length(delays))) .* dw2 * ħ * pi
    y2d = w * ħ
end

function rho0(e, T)
    beta = 1/(kb*T)
    Z = exp.(-beta .* e)
    Z./= sum(Z)
    return diagm(0=>Z)
end

function linear_spectrum(temperature, e, mu, probeT, Es) 
    T = 5*probeT
    y0 = NLSolve.steady_state(rho0(e, temperature), T)
    G! = NLSolve.make_G!(e, ħ=ħ)
    out = []
    out_I = []
    NLSolve.V!(y0, mu, ħ=ħ)

    for E in Es
        w0 = E/ħ
        sigma = fwhm2sigma(probeT)
        et, ew = make_gaussian_field(1.0, w0, sigma)
        wmin = w0-5*sigma
        wmax = w0+5*sigma
        probe = NLSolve.Interaction([wmin], [wmax], [ew], [false])
        field = NLSolve.compile_interaction(probe, y0)

        # println("computing y1...")
        y1 = NLSolve.E(y0, field)
        y1 = G!(y1)

        # println("computing spectrum...")
        w,ta = NLSolve.take_spectrum(y1, probe, mu)
        dw = w[2] - w[1]
        sig = sum(ta) * dw
        iin = sum(abs.(ew.(w)).^2) * dw

        println("E: ", E, " Abs:", imag(sig))
        append!(out, sig)
        append!(out_I, iin)
    end
    return vcat(out...), vcat(out_I...)
end



function cars_spectrum(temperature, e, mu, fwhm, wpump, Es;
                       full=false, N=4) 

    # w1 - w2 + w1 -> 2 * w1 - w2

    T = N*fwhm
    sigma = fwhm2sigma(fwhm)
    y0 = NLSolve.steady_state(rho0(e, temperature), T)
    G! = NLSolve.make_G!(e, ħ=ħ)

    w1min = wpump - 5*sigma
    w1max = wpump + 5*sigma
    et1, ew1 = make_gaussian_field(1.0, wpump, sigma)
    f1 = NLSolve.Interaction([w1min], [w1max], [ew1], [false])

    out = []
    out_s = []
    out_w = []

    for E in Es
        wstokes = wpump - E/ħ
        wmin = wstokes-5*sigma
        wmax = wstokes+5*sigma
        et2, ew2 = make_gaussian_field(1.0, wstokes, sigma)
        f2 = NLSolve.Interaction([wmin], [wmax], [ew2], [false])

        wsignal = 2 * wpump - wstokes
        window_min1 = wsignal - 5 * sigma
        window_max1 = wsignal + 5 * sigma
        window1(w) = (w > window_min1) & (w < window_max1)

        window_min2 = wsignal - 3 * sigma
        window_max2 = wsignal + 3 * sigma
        window2(w) = (w > window_min2) & (w < window_max2)

        fc1 = NLSolve.compile_interaction(f1, y0)
        fc2 = NLSolve.compile_interaction(conj(f2), y0)

        y = NLSolve.ket_deepcopy(y0)
        # pt 1 (pump)
        y = NLSolve.V!(y, mu, ħ=ħ)
        y = NLSolve.E(y, fc1)
        y = G!(y)

        # pt 2 (dump with stokes)
        y = NLSolve.V!(y, mu, ħ=ħ)
        y = NLSolve.E(y, fc2)
        y = G!(y)
        
        # pt 3 (apply pump beam again)
        y = NLSolve.V!(y, mu, ħ=ħ)

        if full
            y = NLSolve.E(y, fc1)
        else
            y = NLSolve.E(y, fc1, filter=x-> window1(x))
        end
        y = G!(y)

        # mu
        w, Esig = NLSolve.take_signal(y, mu)

        Isig = abs.(Esig).^2
        cars = sum(Isig[window2.(w)]) * y0.dω

        push!(out, cars)
        push!(out_s, Esig)
        push!(out_w, w)
        println("E: ", E, " signal:", cars)
    end
    return vcat(out...), out_w, out_s
end

function pump_probe(temperature, e, mu,
                    τs, T,
                    wpump, FWHMpump,
                    wprobe, FWHMprobe; probe_chirp=0.0)

    y0 = NLSolve.steady_state(rho0(e, temperature), T)
    G! = NLSolve.make_G!(e, ħ=ħ)

    # pump
    w0 = wpump
    sigma = fwhm2sigma(FWHMpump)
    et, ew = make_gaussian_field(1.0, w0, sigma)
    wmin = w0-4*sigma
    wmax = w0+4*sigma
    pump = NLSolve.Interaction([wmin], [wmax], [ew], [false])
    cpump = NLSolve.compile_interaction(pump + conj(pump), y0)
    println(cpump.lls)
    println(cpump.uls)

    y2 = let f=cpump, mu=mu, y0=NLSolve.ket_deepcopy(y0)
        y1 = NLSolve.V!(y0, mu, ħ=ħ)
        y1 = NLSolve.E(y1, cpump)
        y1 = G!(y1)

        y2 = NLSolve.V!(y1, mu, ħ=ħ)
        y2 = NLSolve.E(y2, cpump, filter=x -> abs(x)<2.0/ħ)
        y2 = G!(y2)

        NLSolve.V!(y2, mu, ħ=ħ)
    end

    out = []
    # force the GCs hand
    GC.gc(true)

    function window(x, wmin, wmax)
        if (x>=wmin) & (x<=wmax)
            return true
        else
            return false
        end
    end

    w0 = wprobe
    sigma = fwhm2sigma(FWHMprobe)
    A = 20.0

    for τ in τs
        et, ew = make_gaussian_field(A, w0, sigma, delay=-τ, chirp=probe_chirp)
        wmin = w0-4*sigma
        wmax = w0+4*sigma
        probe = NLSolve.Interaction([wmin], [wmax], [ew], [false])
        cprobe = NLSolve.compile_interaction(probe, y0)
        
        println(cprobe.lls)
        println(cprobe.uls)
        # println("y3")
        y3 = NLSolve.E(y2, cprobe, filter=x->window(x,wmin,wmax))
        y3 = G!(y3)

        w,spec = NLSolve.take_spectrum(y3, probe, mu)
        push!(out, spec)
        println("tau: ", τ, " min(spec):", minimum(imag(spec)))
    end

    et, ew = make_gaussian_field(A, w0, sigma, chirp=probe_chirp)
    wmin = w0-4*sigma
    wmax = w0+4*sigma
    probe = NLSolve.Interaction([wmin], [wmax], [ew], [false])
    cprobe = NLSolve.compile_interaction(probe, y0)

    y1 = NLSolve.V!(y0, mu, ħ=ħ)
    y1 = NLSolve.E(y1, cprobe)
    y1 = G!(y1)
    
    w,spec0 = NLSolve.take_spectrum(y1, probe, mu)
    

    return w, hcat(out...), spec0, ew.(w)
end


e2, mu2 = load("pyr2.jld", "e", "mu")
mu2 = Complex.(mu2)

do_linear = true
do_cars = true
do_TA = true

output_folder = "output/pyrazine/"

if do_linear
    # Linear absorption examples
    El = 3.7 
    Eu = 5.5 
    Es = collect(range(El, stop=Eu, step=0.001))
    abs2_300fs, I2_300 = linear_spectrum(300.0, e2, mu2, 300.0, Es)
    abs2_1000fs, I2_1000 = linear_spectrum(300.0, e2, mu2, 1000.0, Es)
    f = string(output_folder, "linear.jld")
    save(f,
         "E", Es,
         "abs2_300fs",  abs2_300fs ./ I2_300,
         "abs2_1000fs", abs2_1000fs ./ I2_1000)
end

if do_cars
    Es = collect(0.05:0.001:0.09)
    Scars2, ws, Isigs = cars_spectrum(300.0, e2, mu2, 2000.0, 4.6/ħ, [0.054],full=true) 
end

if do_TA
    taus = collect(-100.0:1.0:300.0)

    w1, Spp1, Sprobe1, Eprobe1 = pump_probe(300.0, e2, mu2 * 0.1,
                    taus, 600.0,
                    4.8/ħ, 20.0,
                    4.3/ħ,  5.0)

    taus2 = collect(-200.0:1.0:600.0)
    w2, Spp2, Sprobe2, Eprobe2 = pump_probe(300.0, e2, mu2 * 0.1,
                    taus2, 1200.0,
                    4.8/ħ, 40.0,
                    4.8/ħ, 10.0)

    w3, Spp3, Sprobe3, Eprobe3 = pump_probe(300.0, e2, mu2 * 0.1,
                    taus2, 1200.0,
                    4.8/ħ, 40.0,
                    4.8/ħ, 20.0)

    w4, Spp4, Sprobe4, Eprobe4 = pump_probe(300.0, e2, mu2 * 0.1,
                    taus2, 1200.0,
                    4.8/ħ, 40.0,
                    4.2/ħ, 10.0, probe_chirp=500.0)

    
    save(f,
         "tau1",  taus,
         "E1", w1 * ħ,
         "Spp1", Spp1,
         "Sprobe1", Sprobe1,
         "Eprobe1", Eprobe1,
         "tau2",  taus2,
         "E2", w2 * ħ,
         "Spp2", Spp2,
         "Sprobe2", Sprobe2,
         "Eprobe2", Eprobe2,
         "tau3",  taus2,
         "E3", w3 * ħ,
         "Spp3", Spp3,
         "Sprobe3", Sprobe3,
         "Eprobe3", Eprobe3,
         "tau4",  taus2,
         "E4", w4 * ħ,
         "Spp4", Spp4,
         "Sprobe4", Sprobe4,
         "Eprobe4", Eprobe4,
         )
end

