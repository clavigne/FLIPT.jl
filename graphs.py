import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import h5py
import scipy.signal 

from scipy.integrate import ode

def HO(w, l, E, tf, dt, y0):
    N = y0.shape[1]
    def f(t, y):
        out = np.zeros((2,N))
        inp = y.reshape((2,N))
        out[0,:] = inp[1,:] * w
        out[1,:] = -inp[0,:] * w - l
        return out.flatten()

    r = ode(f).set_integrator("vode")
    r.set_initial_value(y0.flatten())
    ts = [0.0]
    ys = [y0]
    while r.successful() and r.t < tf/hbarfs:
        ys += [r.integrate(r.t + dt/hbarfs).reshape((2,N))]
        ts += [r.t * hbarfs]

    ts = np.array(ts)
    ys = np.array(ys)
    print ys.shape
    es = w*(ys[:,0,:]**2.0)/2.0 + l * ys[:,0,:] + E
    return ts, es

def absorbance(Ihet, Iin):
    # Compute absorbance from the heterodyne signal and field intensity.

    # The heterodyne signal is given by
    #     Ihet = Ein^* Esig \approx |Ein + Esig|^2 - |Ein|^2

    # transmission
    T = 1.0 + np.imag(Ihet)/Iin

    # absorbance
    return -np.log10(T)
    


ev_nm = 1239.842
ev_wn = 8065.54
planck = 4.13566751691e-15 # ev s
hbarfs = planck * 1e15 / (2 * np.pi) #ev fs

def takeout(f, name):
    if name in f:
        return f[name].value
    elif name + "_r" in f:
        return f[name+"_r"].value + 1j * f[name + "_i"].value
    else:
        raise KeyError("Dataset %s does not exist" % name)

pyr_linear = False
pyr_cars = False
pyr_ta = False
pyr_ta_chirp = False
lam_pops = True
lam_time = True

if lam_time:
    f = h5py.File("lambda_data.h5", "r")
    alpha = takeout(f, "scal/scales") 
    fwhm = takeout(f, "scal/FWHM") 
    T = takeout(f, "scal/T") 

    flipt = np.sum(takeout(f, "scal/flipt"), 1)
    direct = takeout(f, "scal/direct")

    npts = takeout(f, "abs/npoints")[:,0]
    which = (npts<2800) & (npts>400)
    npts = npts[which]
    flipt2 = takeout(f, "abs/flipt")[which,:]


    fig, axes = plt.subplots(1, 1, figsize=(3.375, 3.375*0.75), dpi=200)
    plt.sca(axes)
    plt.plot(T, flipt, 'x', c="blue", alpha=0.2)
    plt.plot(T, direct, 'x', c="gray", alpha=0.5)
    plt.ylabel("Run time (s)")
    plt.xlabel("Propagation time (fs)")
    plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.18)
    plt.savefig("graphs/lam/timings.pdf")


    fig, axes = plt.subplots(1, 1, figsize=(3.375, 3.375*0.75), dpi=200)
    plt.sca(axes)
    plt.loglog(npts, flipt2, 'x', alpha=0.2)

    p = np.polyfit(np.log(npts), np.log(flipt2),1)
    x = sorted(list(set(npts)))

    def anot_fit(order):
        y = np.exp(p[1,order-1]) * x**p[0, order-1]
        plt.loglog(x, y, 'k:', linewidth=0.5)
        plt.annotate(r"$\alpha^{(%i)} =%3.1f$"% (order, p[0,order-1]),
                    xy = (x[-1], y[-1]),
                    xytext = (10, 0), textcoords="offset points",
                    fontsize=8)
    anot_fit(1)
    anot_fit(2)
    anot_fit(3)
    anot_fit(4)
    axdims = plt.axis("tight")
    plt.axis([axdims[0],axdims[1]*2,axdims[2],axdims[3]*2])


    plt.ylabel(r"Run time (s)")
    # axes[1].axis([-20.0, 300.0, 0.0, 2e-5])
    # plt.yticks([0.0, 1e-5, 2e-5], [r"0",r"$1$", r"$2$"])
    plt.xlabel("Number of points")

    plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.18)
    plt.savefig("graphs/lam/timings_orders.pdf")

    f.close()

if lam_pops:
    f = h5py.File("lambda_data.h5", "r")
    t = takeout(f, "short/times") 
    yf = takeout(f, "short/flipt") 
    yd = takeout(f, "short/direct") 
    y1f = yf[:,1,1,2].real + yf[:,2,2,2].real
    y1d = yd[:,1,1,2].real + yd[:,2,2,2].real

    y2f = yf[:,1,1,4].real + yf[:,2,2,4].real
    y2d = yd[:,1,1,4].real + yd[:,2,2,4].real

    fig, axes = plt.subplots(2, 1, figsize=(3.375, 3.375*1.5), dpi=200, sharex=True)
    plt.sca(axes[0])
    plt.plot(t, y1f, "k")
    plt.plot(t, y1d, "k:")
    plt.axis([-20.0, 300.0, 0.0, 10e-3])
    plt.yticks([0.0, 5e-3, 10e-3], [r"0",r"$5$", r"$10$"])
    plt.xticks([0, 100, 200, 300])
    plt.ylabel(r"$\rho^{(2)}_{ee} (\times 10^{-3})$")

    axins = inset_axes(axes[0], width="40%", height="40%", loc=4, borderpad=2)
    plt.sca(axins)
    plt.semilogy(t, abs(y1d-y1f)/abs(y1f), "k", linewidth=1)
    plt.axis([0.0, 300.0, 1e-5, 1e-3])
    plt.yticks([1e-5, 1e-4, 1e-3])
    plt.grid(True)
    plt.xticks([0, 100, 200, 300])
    axins.tick_params(labelleft=True)

    plt.sca(axes[1])
    axes[1].plot(t, -y2f, "k")
    axes[1].plot(t, -y2d, "k:")
    axes[1].axis([-20.0, 300.0, 0.0, 10e-5])
    plt.yticks([0.0, 5e-5, 10e-5], [r"0",r"$5$", r"$10$"])
    plt.xlabel("Time (fs)")
    plt.ylabel(r"-$\rho^{(4)}_{ee} (\times 10^{-5})$")

    axins = inset_axes(axes[1], width="40%", height="40%", loc=4, borderpad=2)
    plt.sca(axins)
    plt.semilogy(t, abs(y2d-y2f)/abs(y2f), "k", linewidth=1)
    plt.axis([0.0, 300.0, 1e-3, 1e-1])
    plt.yticks([1e-3, 1e-2, 1e-1])
    plt.grid(True)
    plt.xticks([0, 100, 200, 300])
    axins.tick_params(labelleft=True)

    plt.sca(axes[1])
    plt.subplots_adjust(left=0.16, right=0.95, top=0.95)
    plt.savefig("graphs/lam/short_exc.pdf")

    f.close()

if pyr_linear:
    f = h5py.File("pyrazine_data.h5", "r")
    # linear spectra
    x = ev_nm/f['linear/E'].value
    names = ["abs2_300fs", "abs2_1000fs", "abs4_300fs", "abs4_1000fs"]
    data = np.array([takeout(f, "linear/" + s) for s in names])

    def linear_spectra(name, data1, data2):
        fig, ax = plt.subplots(figsize=(3.375*2, 3.375), dpi=200)
        plt.plot(x, np.imag(data1), "k-")
        # plt.xlabel("Energy $\hbar \omega$ (eV)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absorbance (a.u.)")
        axes = plt.axis("tight")
        plt.axis([225, 280, axes[2], axes[3]])

        axins = inset_axes(ax, width="30%", height="30%", loc=2)
        plt.sca(axins)
        plt.plot(x, np.imag(data2), "k--")
        plt.plot(x, np.imag(data1), "k")
        plt.axis([245, 251, axes[2],axes[3]/2.0])
        axins.tick_params(labelleft=False, labelright=False)

	plt.subplots_adjust(bottom=0.2)
        plt.savefig("graphs/linear/" + name +".pdf")

    linear_spectra("2modes", data[0], data[1])
    linear_spectra("4modes", data[2], data[3])
    f.close()

if pyr_cars:
    def ef(x, data, vib, ket, peakwidth=6):
        energy = sum([vib[m]*ket[m] for m in range(len(ket))])
        ind=np.searchsorted(x, energy)
        if ind == len(x):
            print str(ket) + " > than max e" 
        else:
            ind2 = np.argmax(data[ind-peakwidth:ind+peakwidth])
            loc = x[ind2+ind-peakwidth]
            yloc = data[ind2 +ind-peakwidth]
            vs = reduce(lambda x,y:x+y, [("%i"%v) for v in ket])
            return vs, loc, yloc

    def cars_spectra(name, x, data, wvib, phonons): 
        nmodes = len(wvib)
        plt.figure(figsize=(3.375*2, 3.375), dpi=200)
        plt.plot(x, data, 'k')

        for ket in phonons:
            vs, loc, yloc = ef(x, data, wvib, ket)
            plt.annotate(vs, (loc,
                              yloc), (-2.5 * nmodes, 10), textcoords="offset points")



        ax = plt.axis('tight')
        # A = np.max(data[xmin:xmax])
        # plt.axis([500.0, 1930.0,
        #           -A/5.0, A * 1.2])
        plt.xlabel("Energy $\hbar \omega$ (eV)")
        # plt.xlabel("Frequency (cm$^{-1}$)")
        plt.ylabel("CARS signal (a.u.)")
	plt.subplots_adjust(bottom=0.2)
        plt.savefig("graphs/cars/" + name + ".pdf")

    # four modes
    quantas4 = [0.0739 , 0.1139 , 0.1258 , 0.1525 ]
    kets4 = [(1,0,0,0), (1,1,0,0), (1,0,1,0),
             (1,0,0,1), (2,0,0,0), (0,1,0,0),
             (0,0,1,0)]
    quantas2 = [0.0739 , 0.1139]
    kets2 = [(1,0), (0,1), (2,0), (0,2), (1,1)]

    # cars spectra
    f = h5py.File("pyrazine_data.h5", "r")
    x = f['cars/E'].value
    data = f['cars/Scars2']
    cars_spectra("2modes", x, data, quantas2, kets2)


    # # Ultrafast cars
    # f = h5py.File("pyrazine_data.h5", "r")
    # x = f['pulsed_cars/E_200'].value
    # data = f['pulsed_cars/Icars_200'].value
    # cars_spectra("2modes_ultrafast", x-90/ev_wn, data, quantas2, kets2)


    # f = h5py.File("pyrazine_data.h5", "r")
    # x = f['pulsed_cars/E_2000'].value
    # data = f['pulsed_cars/Icars_2000'].value
    # cars_spectra("2modes_ultrafast_2000", x - 90/ev_wn, data, quantas2, kets2)

    f.close()

if pyr_ta:
    f = h5py.File("pyrazine_data.h5", "r")
    which = "1"

    tau = takeout(f, "ta/tau" + which)
    E = takeout(f, "ta/E" + which)
    Spp = takeout(f, "ta/Spp" + which)
    Sprobe = takeout(f, "ta/Sprobe" + which)
    Eprobe = takeout(f, "ta/Eprobe" + which)
    Iprobe = abs(Eprobe)**2

    Ap = absorbance(Sprobe, Iprobe)
    App = absorbance(Spp + Sprobe, Iprobe)
    dAbs = Ap-App 
    lim = np.max(abs(dAbs))

    # -------------------- Transient absorption -----------------
    plt.figure(figsize=(3.375, 3.375), dpi=200)
    plt.pcolormesh(tau, E, dAbs.T, cmap=plt.cm.seismic, vmin=-lim, vmax=lim,
                   rasterized=True)

    plt.axis([-50, np.max(tau), 3.6, 5.5])
    plt.xlabel(r"$\tau$ (fs)")
    plt.ylabel(r"Probe energy (eV)")
    plt.subplots_adjust(left=0.20, bottom=0.12)
    plt.savefig("graphs/ta/ta.pdf")


    # -------------------- CI -----------------
    plt.figure(figsize=(3.375, 3.375), dpi=200)
    plt.pcolormesh(tau, E, dAbs.T, cmap=plt.cm.seismic, vmin=-lim/2, vmax=lim/2,
                   rasterized=True)

    qci = -4.06834825061
    w = 0.0739
    y2 = lambda x: w * x**2.0/2.0 + 0.13545 * x + 4.89
    y1 = lambda x: w * x**2.0/2.0 - 0.09806 * x + 3.94

    q = tau * w/hbarfs
    shift = 0 * 2 * np.pi * hbarfs/w
    v2 = y2(-q)
    v1 = y1(q + 2 * qci)
    v2p = y2(q + 2 * qci)
    tci = np.argmin(abs(-q - qci))

    plt.plot(tau[:tci] + shift, v2[:tci], "k", linewidth=3)
    plt.plot(tau[tci:] + shift, v1[tci:], "k:", linewidth=3)
    plt.plot(tau[tci:] + shift, v2[tci:], "k", linewidth=3)
    # plt.plot(tau[tci:] + shift, v2p[tci:], "k:", linewidth=3)

    ci_coords = (-qci * hbarfs/w + shift, y2(qci))
    plt.plot(ci_coords[0], ci_coords[1], "ko", markersize=10)
    plt.annotate("CI", xy=ci_coords, size=15,
                 xytext=(-15, 10), textcoords="offset points")

    plt.annotate(r"$v_2(\tau)$", xy=(53, 5.4), size=11,
                 xytext=(10, 0), textcoords="offset points")
    
    plt.annotate(r"$v_1(\tau)$", xy=(50, 4.5), size=11,
                 xytext=(0, 0), textcoords="offset points")


    plt.axis([-10.0, 80.0, 4.4, 5.5])
    plt.xlabel(r"$\tau$ (fs)")
    plt.ylabel(r"Probe energy (eV)")
    plt.subplots_adjust(left=0.20, bottom=0.12)
    plt.savefig("graphs/ta/ta_CI.pdf")

    # # -------------------- Pump-probe --------------------------
    # plt.figure(figsize=(3.375, 3.375), dpi=200)
    # dE = E[1] - E[0]
    # wS2 = E>4.25
    # pump_probe_2 = np.sum(np.imag(Spp[:, wS2]), 1) * dE
    # N = np.max(abs(pump_probe_2))

    # plt.plot(tau, pump_probe_2/N, 'k')

    # plt.xlabel(r"$\tau$ (fs)")
    # plt.ylabel(r"$S_{pp}(\tau)$ (a.u.)")
    # plt.axis([-50.0, 300.0, 0.0, 1.1])
    # plt.subplots_adjust(left=0.20, bottom=0.12)
    # plt.savefig("graphs/ta/pump_probe.pdf")
    # f.close()


