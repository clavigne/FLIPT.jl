using JLD
using HDF5
function takeout(folder, fname, vars)
    f = string(folder, fname)
    Dict((var, load(f, var)) for var in vars)
end

function putin(file, data_id,  v::AbstractArray{T, N}) where {T<:Complex, N}
    write(file, string(data_id, "_r"), real(v))  
    write(file, string(data_id, "_i"), imag(v))  
end

function putin(file, data_id, v) 
    write(file, data_id, v)
end

do_pyrazine = true
do_lambda = true

if do_pyrazine
    folder = "output/pyrazine/"
    linear = takeout(folder, "linear.jld", ["E",
                                            "abs2_300fs",
                                            "abs2_1000fs"])
    cars = takeout(folder, "cars.jld", ["E",
                                        "Scars2"])
    ta = takeout(folder, "ta.jld",
                 [
                  "tau1", "E1", "Spp1", "Sprobe1", "Eprobe1",
                  "tau2", "E2", "Spp2", "Sprobe2", "Eprobe2",
                  "tau3", "E3", "Spp3", "Sprobe3", "Eprobe3",
                  "tau4", "E4", "Spp4", "Sprobe4", "Eprobe4",
                  ])
    pulsed_cars = takeout(folder, "cars_ultrafast.jld",
                          ["E_200", "Icars_200",
                           "E_2000", "Icars_2000"])

    h5open("pyrazine_data.h5", "w") do file
        for (k,v) in linear
            putin(file, string("linear/", k), v)
        end

        for (k,v) in ta
            putin(file, string("ta/", k), v)
        end

        for (k,v) in cars
            putin(file, string("cars/", k), v)
        end

        for (k,v) in pulsed_cars
            putin(file, string("pulsed_cars/", k), v)
        end
    end
end

if do_lambda
    folder = "output/compare/"
    short = takeout(folder, "short.jld", ["times", "flipt","direct"])
    long = takeout(folder, "long.jld", ["times", "flipt", "direct"])
    scal = takeout(folder, "scaled_timings.jld", ["scales", "FWHM", "T", "flipt", "direct"])
    abs = takeout(folder, "absolute_timings.jld", ["flipt", "npoints"])

    h5open("lambda_data.h5", "w") do file
        for (k,v) in short
            putin(file, string("short/", k), v)
        end

        for (k,v) in long
            putin(file, string("long/", k), v)
        end

        for (k,v) in scal
            putin(file, string("scal/", k), hcat(v...))
        end

        for (k,v) in abs
            putin(file, string("abs/", k), hcat(v...))
        end
    end
end

