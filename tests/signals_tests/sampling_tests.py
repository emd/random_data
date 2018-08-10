from nose import tools
import numpy as np
from scipy.fftpack import next_fast_len
import random_data as rd


def test_circular_resample():
    # Complex input should raise ValueError:
    # ======================================
    x = np.zeros(10, dtype='complex')
    tools.assert_raises(
        ValueError,
        rd.signals.sampling.circular_resample,
        *[x, 1., 0])

    # Null (i.e. no shift):
    # =====================

    # N even, 5-smooth number:
    # ------------------------
    N = 10
    x = np.arange(N)
    Fs = 1.
    tau = 0

    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(xr, x)

    # N even, *not* a 5-smooth number:
    # --------------------------------
    N = 14
    x = np.arange(N)
    Fs = 1.
    tau = 0

    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(xr, x)

    # N odd, 5-smooth number:
    # -----------------------
    N = 9
    x = np.arange(N)
    Fs = 1.
    tau = 0

    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(xr, x)

    # N odd, *not* a 5-smooth number:
    # -------------------------------
    N = 11
    x = np.arange(N)
    Fs = 1.
    tau = 0

    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(xr, x)

    # Positive, integer shift:
    # ========================

    # N even, 5-smooth number:
    # ------------------------
    N = 10
    x = np.arange(N)
    Fs = 1.
    tau = 1

    # No zero padding of `x`, so simply moves `x[0]` to end of array
    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        np.concatenate((x[1:], [x[0]])))

    # N even, *not* a 5-smooth number:
    # --------------------------------
    N = 14
    x = np.arange(N)
    Fs = 1.
    tau = 1

    # Zero padding of `x` means that `x[0]` is cycled out of the
    # viewing window and a `0` is cycled into the end of `x`
    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        np.concatenate((x[1:], [0])))

    # N odd, 5-smooth number:
    # -----------------------
    N = 9
    x = np.arange(N)
    Fs = 1.
    tau = 1

    # No zero padding of `x`, so simply moves `x[0]` to end of array
    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        np.concatenate((x[1:], [x[0]])))

    # N odd, *not* a 5-smooth number:
    # -------------------------------
    N = 11
    x = np.arange(N)
    Fs = 1.
    tau = 1

    # Zero padding of `x` means that `x[0]` is cycled out of the
    # viewing window and a `0` is cycled into the end of `x`
    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        np.concatenate((x[1:], [0])))

    # Negative, integer shift:
    # ========================

    # N even, 5-smooth number:
    # ------------------------
    N = 10
    x = np.arange(N)
    Fs = 1.
    tau = -1

    # No zero padding of `x`, so simply moves `x[-1]` to start of array
    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        np.concatenate(([x[-1]], x[:-1])))

    # N even, *not* a 5-smooth number:
    # --------------------------------
    N = 14
    x = np.arange(N)
    Fs = 1.
    tau = -1

    # Zero padding of `x` means that `x[-1]` is cycled out of the
    # viewing window and a `0` is cycled into the start of `x`
    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        np.concatenate(([0], x[:-1])))

    # N odd, 5-smooth number:
    # -----------------------
    N = 9
    x = np.arange(N)
    Fs = 1.
    tau = -1

    # No zero padding of `x`, so simply moves `x[-1]` to start of array
    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        np.concatenate(([x[-1]], x[:-1])))

    # N odd, *not* a 5-smooth number:
    # -------------------------------
    N = 11
    x = np.arange(N)
    Fs = 1.
    tau = -1

    # Zero padding of `x` means that `x[-1]` is cycled out of the
    # viewing window and a `0` is cycled into the start of `x`
    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        np.concatenate(([0], x[:-1])))

    # Positive, non-integer shift:
    # ============================
    #
    # Note that `circular_resample` uses the FFT, which maps a finite
    # `N`-length sequence onto the unit circle with the implicit assumption
    # of periodic boundary conditions. The resulting periodicity allows
    # representation of the signal with an `N`-term Fourier series.
    #
    # Now, after being mapped onto the unit circle, the ramping sequence
    # used in the above computations (`x = np.arange(N)`) corresponds to a
    # periodic sawtooth signal. It is well-known that the Fourier series
    # does a poor job of representing the underlying signal at sharp edges
    # (i.e. the Gibbs phenomenon), such as the vertices of the sawteeth.
    # The Gibbs phenomenon becomes readily apparent when shifting the
    # sawtooth signal by a non-integer number of samples.
    #
    # Because of this, it makes more sense to use a simpler signal, such
    # as a pure sinusoid with period equal to the sequence length, to check
    # that `circular_resample` is performing as expected for non-integer
    # shifts. Note that this test only works well if we do *not* zero pad
    # the input signal to a 5-smooth number (i.e. `len(x)` needs to already
    # be a 5-smooth number, or the `pad_to_next_fast_len` keyword argument
    # must be `False`).

    def unity_frequency_sine(t):
        'Get unity frequency sine wave.'
        f0 = 1
        return np.sin(2 * np.pi * f0 * t)

    def period_N_sine_wrapper(N):
        'Get `N`-length sine wave containing exactly *one* period.'
        t = np.arange(0, N, dtype='float') / N
        Fs = 1. / (t[1] - t[0])
        x = unity_frequency_sine(t)

        return t, Fs, x

    # N even, 5-smooth number:
    # ------------------------
    N = 10
    t, Fs, x = period_N_sine_wrapper(N)
    tau = 0.5 / Fs

    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        unity_frequency_sine(t + tau))

    # N odd, 5-smooth number:
    # -----------------------
    N = 9
    t, Fs, x = period_N_sine_wrapper(N)
    tau = 0.5 / Fs

    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        unity_frequency_sine(t + tau))

    # Negative, non-integer shift:
    # ============================

    # N even, 5-smooth number:
    # ------------------------
    N = 10
    t, Fs, x = period_N_sine_wrapper(N)
    tau = -0.5 / Fs

    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        unity_frequency_sine(t + tau))

    # N odd, 5-smooth number:
    # -----------------------
    N = 9
    t, Fs, x = period_N_sine_wrapper(N)
    tau = -0.5 / Fs

    xr = rd.signals.sampling.circular_resample(
        x, Fs, tau, pad_to_next_fast_len=True)

    np.testing.assert_almost_equal(
        xr,
        unity_frequency_sine(t + tau))

    # Spectrum for non-integer shift, N *not* a 5-smooth number:
    # ==========================================================

    # Construct signal:
    # -----------------
    # Desired number of points in signal; *not* a 5-smooth number
    N = 63331
    tools.assert_not_equal(
        N,
        next_fast_len(N),
        msg='`N` *is* a 5-smooth number; testing requires non-5-smooth number')

    # Translate `N` into values usable by `random_data` routines
    N_power_of_2 = np.int(2 ** np.ceil(np.log2(N)))
    Fs = 1.
    t0 = 0
    T = N_power_of_2 / Fs

    # Parameters of broadband spectrum
    f0_broad = 0.
    tau_broad = 2.
    G0 = 1.
    noise_floor = 1e-2
    seed = None

    # Broadband signal, where `len(sig.x)` is a power of 2
    sig = rd.signals.RandomSignal(
        Fs=Fs, t0=t0, T=T,
        f0=f0_broad, tau=tau_broad, G0=G0,
        noise_floor=noise_floor, seed=seed)

    # Only take the first `N` points of `sig.x`, producing
    # a signal with a length that is *not* a 5-smooth number
    x = sig.x[:N]
    t = sig.t()[:N]

    tools.assert_not_equal(
        len(x),
        next_fast_len(len(x)),
        msg='`len(x)` *is* a 5-smooth number; non-5-smooth number required')

    # Circularly resample signal:
    # ---------------------------
    # Integer shift
    tau_1 = 1. / Fs
    x_1 = rd.signals.sampling.circular_resample(
        x, Fs, tau_1, pad_to_next_fast_len=True)

    # Non-integer shift
    tau_05 = 0.5 / Fs
    x_05 = rd.signals.sampling.circular_resample(
        x, Fs, tau_05, pad_to_next_fast_len=True)

    # Compare the spectra of the original and time-shifted signals:
    # -------------------------------------------------------------
    # Spectral-estimation parameters
    Nreal_per_ens = 1000
    Tens = t[-1] - t[0]  # Define ensemble to be full length of `x`

    # Estimate spectra
    asd = rd.spectra.AutoSpectralDensity(
        x, Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    asd_1 = rd.spectra.AutoSpectralDensity(
        x_1, Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    asd_05 = rd.spectra.AutoSpectralDensity(
        x_05, Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Relative error of time-shifted spectra relative to original
    relerr_1 = np.squeeze((asd_1.Gxx - asd.Gxx) / asd.Gxx)
    relerr_05 = np.squeeze((asd_05.Gxx - asd.Gxx) / asd.Gxx)

    # Relative random error of an autospectral density estimate
    # is 1 / sqrt(Nreal_per_ens). Check that the relative errors
    # of the time-shifted spectra are "not too much larger" than
    # the random error expected for the unshifted spectral estimate.
    # The "not too much larger" is quantified by `fudge_factor`.
    # Note that a fudge factor of 2 is larger than typically needed,
    # but it ensures robust testing in the presence of rare
    # statistical events that more severely affect the shift
    # calculation.
    fudge_factor = 2.
    max_relerr = fudge_factor / np.sqrt(Nreal_per_ens)

    tools.assert_less_equal(
        np.max(np.abs(relerr_1)),
        max_relerr)

    tools.assert_less_equal(
        np.max(np.abs(relerr_05)),
        max_relerr)

    return


def test_TriggerOffset_2signals():
    # Setup parameters:
    # -----------------
    # Nominal temporal-grid parameters
    Fs = 4e6                        # [Fs] = samples / s
    t0 = 0                          # [t0] = s
    T = 1e5 / Fs                    # [T] = s

    # Discrepancy `tau` between timebase of digital record 1 and
    # timebase of digital record 2. Note that the accuracy of
    # the trigger-offset estimation improves with larger `tau`, so
    # use a relatively large value for automated testing here.
    tau = 10e-6                     # [tau] = s, nonzero

    # The number of timestamps by which digital record 2 should be shifted
    # to probe the timebase discrepancy `tau`
    shifts = np.arange(-10, 11, 1)

    # Spectral parameters of broadband signal common to digital records 1 & 2
    f0_broad = 0.
    tau_broad = 2. / Fs
    G0 = 1.
    noise_floor = 1e-12
    seed = None

    # Spectral-estimation parameters
    Nreal_per_ens = 1000            # [Nreal_per_ens] = unitless
    gamma2xy_max = 0.95             # [gamma2xy_max] = unitless

    # Tests:
    # ------
    sig = rd.signals.RandomSignal(
        Fs=Fs, t0=t0, T=T,
        f0=f0_broad, tau=tau_broad, G0=G0,
        noise_floor=0., seed=seed)
    N = len(sig.x)

    # Generate two digital records that are identical *except* for
    # timebase offset `tau`. Note that `x1[0]` physically occurs at
    # time `t[0]`, while `x2[0]` physically occurs at time `t[0] + tau`.
    x1 = sig.x.copy()
    x2 = rd.signals.sampling.circular_resample(sig.x, Fs, tau)

    # Add uncorrelated noise to generate two distinct signals
    noise_amplitude = np.sqrt(0.5 * Fs * noise_floor)
    x1 += (noise_amplitude * np.random.randn(N))
    x2 += (noise_amplitude * np.random.randn(N))

    # Estimate trigger offset
    trig = rd.signals.TriggerOffset(
        np.array([x1, x2]),
        shifts=(np.int(tau * Fs) + shifts),
        gamma2xy_max=gamma2xy_max,
        Fs=Fs,
        Nreal_per_ens=Nreal_per_ens)

    # Check that estimated trigger offset is within 5% of true offset.
    # (The algorithm often provides estimates that are within a much
    # tighter tolerance than 5%, but we use this as a roof to account
    # for the occasional statistical variation that results in larger
    # errors).
    tools.assert_almost_equal(
        (trig.tau - tau) / tau,
        0.,
        places=1)

    # Now, use estimated value to compensate for trigger offset
    x2_corrected = rd.signals.sampling.circular_resample(x2, Fs, -trig.tau)

    # The trigger offset between `x1` and `x2_corrected` should be minimal
    trig_corrected = rd.signals.TriggerOffset(
        np.array([x1, x2_corrected]),
        shifts=shifts,
        gamma2xy_max=gamma2xy_max,
        Fs=Fs,
        Nreal_per_ens=Nreal_per_ens)

    # Trigger offset of corrected signals is at least 200x smaller than
    # original, uncorrected trigger offset
    tools.assert_almost_equal(
        trig_corrected.tau / trig.tau,
        0,
        places=2)

    return


def test_TriggerOffset_4signals():
    # Setup parameters:
    # -----------------
    # Nominal temporal-grid parameters
    Fs = 4e6                        # [Fs] = samples / s
    t0 = 0                          # [t0] = s
    T = 1e5 / Fs                    # [T] = s

    # Discrepancy `tau` between timebase of digital record 1 and
    # timebase of digital record 2. Note that the accuracy of
    # the trigger-offset estimation improves with larger `tau`, so
    # use a relatively large value for automated testing here.
    tau = 10e-6                     # [tau] = s, nonzero

    # The number of timestamps by which digital record 2 should be shifted
    # to probe the timebase discrepancy `tau`
    shifts = np.arange(-10, 11, 1)

    # Spectral parameters of broadband signal common to digital records
    f0_broad = 0.
    tau_broad = 2. / Fs
    G0 = 1.
    noise_floor = 1e-12
    seed = None

    # Spectral-estimation parameters
    Nreal_per_ens = 1000            # [Nreal_per_ens] = unitless
    gamma2xy_max = 0.95             # [gamma2xy_max] = unitless

    # Tests:
    # ------
    # Common broadband signal
    sig = rd.signals.RandomSignal(
        Fs=Fs, t0=t0, T=T,
        f0=f0_broad, tau=tau_broad, G0=G0,
        noise_floor=0., seed=seed)
    N = len(sig.x)

    xhat = np.fft.rfft(sig.x)
    f = np.fft.rfftfreq(len(sig.x), d=(1. / Fs))

    # Let physical signal have a transit time `tau0` from
    # one transducer to the next, adjacent transducer.
    # Select `tau0` such that the maximum, absolute cross
    # phase from one transducer to the next is < pi.
    tau0 = 1. / Fs
    theta0 = 2 * np.pi * f * tau0
    x1 = np.real(np.fft.irfft(xhat * np.exp(1j * 0 * theta0)))
    y1 = np.real(np.fft.irfft(xhat * np.exp(1j * 1 * theta0)))
    x2 = np.real(np.fft.irfft(xhat * np.exp(1j * 2 * theta0)))
    y2 = np.real(np.fft.irfft(xhat * np.exp(1j * 3 * theta0)))

    # Create additional cross phase between correlation pair `(y1, x2)`
    # by adding a trigger offset `tau` to `x2` and `y2` relative to `x1`
    # and `y1`. Note that `x1[0]` physically occurs at time `t[0]`, while
    # `x2[0]` physically occurs at time `t[0] + tau`. The equivalent
    # statement holds for `y1` and `y2`.
    x2 = rd.signals.sampling.circular_resample(x2, Fs, tau)
    y2 = rd.signals.sampling.circular_resample(y2, Fs, tau)

    # Finally, add uncorrelated noise
    noise_amplitude = np.sqrt(0.5 * Fs * noise_floor)
    x1 += (noise_amplitude * np.random.randn(N))
    y1 += (noise_amplitude * np.random.randn(N))
    x2 += (noise_amplitude * np.random.randn(N))
    y2 += (noise_amplitude * np.random.randn(N))

    # Estimate trigger offset
    trig = rd.signals.TriggerOffset(
        np.array([x1, y1, x2, y2]),
        shifts=(np.int(tau * Fs) + shifts),
        gamma2xy_max=gamma2xy_max,
        Fs=Fs,
        Nreal_per_ens=Nreal_per_ens)

    # Check that estimated trigger offset is within 5% of true offset
    # (The algorithm often provides estimates that are within a much
    # tighter tolerance than 5%, but we use this as a roof to account
    # for the occasional statistical variation that results in larger
    # errors).
    tools.assert_almost_equal(
        (trig.tau - tau) / tau,
        0.,
        places=1)

    # Now, use estimated value to compensate for trigger offset
    x2_corrected = rd.signals.sampling.circular_resample(x2, Fs, -trig.tau)
    y2_corrected = rd.signals.sampling.circular_resample(y2, Fs, -trig.tau)

    # The trigger offset between 1 and 2 should be minimal
    trig_corrected = rd.signals.TriggerOffset(
        np.array([x1, y1, x2_corrected, y2_corrected]),
        shifts=shifts,
        gamma2xy_max=gamma2xy_max,
        Fs=Fs,
        Nreal_per_ens=Nreal_per_ens)

    # Trigger offset of corrected signals is at least 200x smaller than
    # original, uncorrected trigger offset
    tools.assert_almost_equal(
        trig_corrected.tau / trig.tau,
        0,
        places=2)

    return
