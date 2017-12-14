from nose import tools
import numpy as np
import random_data as rd


def test_Bispectrum_signal_input():
    # Different length signals should fail
    x = np.random.randn(np.int(50e3))
    y = np.random.randn(np.int(50e3) + 1)
    tools.assert_raises(ValueError, rd.spectra.Bispectrum, x, y)

    # Complex signals should fail
    xc = x.astype('complex128')
    yc = y[:-1].astype('complex128')
    tools.assert_raises(ValueError, rd.spectra.Bispectrum, xc, y)
    tools.assert_raises(ValueError, rd.spectra.Bispectrum, x, yc)
    tools.assert_raises(ValueError, rd.spectra.Bispectrum, xc, yc)

    return


def test_squared_bicoherence():
    '''Corresponds to Fig. 4 of Kim & Powers, IEEE Trans. Plasma Sci, 1979,
    but use more realizations to reduce random error and the likelihood
    of a false testing fail.

    '''
    # Timebase & frequencies:
    # -----------------------
    # Timebase parameters
    Fs = 2.       # => Nyquist frequency is unity
    Nreal = 1024  # if realizations overlap, we'll end up w/ more than `Nreal`
    Npts_per_real = 128
    t = np.arange(Nreal * Npts_per_real) / Fs

    # Specify wave frequencies
    fb = 0.220  # [fb] = Nyquist frequency
    fc = 0.375  # [fc] = Nyquist frequency
    fd = fb + fc

    # Specify phases of waves b & c
    # -----------------------------
    # Phase of waves b & c, each randomly selected to be within [-pi, pi]
    theta_b = 2 * np.pi * (np.random.rand(1) - 0.5)[0]
    theta_c = 2 * np.pi * (np.random.rand(1) - 0.5)[0]

    # Determine phase for wave d:
    # ---------------------------
    # If wave d is nonlinearly generated via interaction of waves b & c,
    # then wave d will be coherent with waves b & c, i.e.
    #
    #       theta_d = theta_b + theta_c.
    #
    # If wave d is *not* nonlinearly generated via interaction of
    # waves b & c, then wave d will have a random phase relative
    # to waves b & c in *EACH* realization.
    theta_d = np.zeros(len(t))
    for r in np.arange(Nreal):
        i1 = r * Npts_per_real
        i2 = ((r + 1) * Npts_per_real) - 1
        theta_d[i1:i2] = 2 * np.pi * (np.random.rand(1) - 0.5)[0]

    # Construct total signal:
    # -----------------------
    x = np.cos((2 * np.pi * fb * t) + theta_b)
    x += np.cos((2 * np.pi * fc * t) + theta_c)
    x += (0.5 * np.cos((2 * np.pi * fd * t) + theta_d))
    x += (np.cos((2 * np.pi * fb * t) + theta_b)
          * np.cos((2 * np.pi * fc * t) + theta_c))

    # Add noise to signal (interestingly, artifacts *decrease* w/ noise):
    # -------------------------------------------------------------------
    noise_dBc = -20
    Ax_eff = np.std(x)  # Power in signal x is `Ax_eff ^ 2`
    Anoise = np.sqrt(10 ** (noise_dBc / 10.)) * Ax_eff
    x += (Anoise * np.random.randn(len(t)))

    # Bispectral calculations:
    # ------------------------
    B = rd.spectra.Bispectrum(
        x, x, Fs=Fs, t0=t[0],
        tlim=None, Nreal_per_ens=Nreal)

    # Check bispectrum against expected values:
    # -----------------------------------------
    # Nonlinear sum interaction is only responsible
    # for approximately half of the power at (fb + fc).
    # Test that the computed value sits within a reasonable
    # neighborhood of 0.5...
    dfr = np.abs(B.frow - fb)
    dfc = np.abs(B.fcol - fc)
    rind = np.where(dfr == np.min(dfr))[0][0]
    cind = np.where(dfc == np.min(dfc))[0][0]
    tools.assert_greater(B.b2xy[rind, cind], 0.35)
    tools.assert_less(B.b2xy[rind, cind], 0.65)

    # Nonlinear difference interaction is responsible
    # for approximately all of the power at (fc - fb)
    dfr = np.abs(B.frow + fb)
    rind = np.where(dfr == np.min(dfr))[0][0]
    tools.assert_greater(B.b2xy[rind, cind], 0.85)
    tools.assert_less_equal(B.b2xy[rind, cind], 1.0)

    return
