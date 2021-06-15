function fwhm2sigma(fwhm)
    a = 1.0/(2.0 * (fwhm/(2 * sqrt(2 * log(2))))^2)
    return sqrt(2*a)
end

function make_gaussian_field(I, w0, sigma; chirp=0.0, delay=0.0)
    A = sqrt(I)
    prefac = A/sqrt(sqrt(sigma^2 * pi) * 2 )
    ap = 1/(2 * sigma^2) + im * chirp
    an = 1/(2 * sigma^2) - im * chirp        

    a_t = 1/(4 * an)
    p_t = sqrt(a_t/pi) * prefac
    fieldt(t) = 2 .* real(p_t .*
                     exp.((-a_t) .* (t.-delay).^2 .+ (im * w0 ).* (t.-delay)))
    fieldw(w) = prefac .* exp.(-ap .* (w .- w0).^2  .+ im .* w .* delay)
    # todo check sign of delay!
    return fieldt, fieldw
end
