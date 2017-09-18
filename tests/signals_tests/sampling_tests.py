from nose import tools
import numpy as np
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

    # N even:
    # -------
    N = 10
    x = np.arange(N)
    Fs = 1.
    tau = 0

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        x)

    # N odd:
    # ------
    N = 11
    x = np.arange(N)
    Fs = 1.
    tau = 0

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        x)

    # Positive, integer shift:
    # ========================

    # N even:
    # -------
    N = 10
    x = np.arange(N)
    Fs = 1.
    tau = 1

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        np.concatenate((x[1:], [x[0]])))

    # N odd:
    # ------
    N = 11
    x = np.arange(N)
    Fs = 1.
    tau = 1

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        np.concatenate((x[1:], [x[0]])))

    # Negative, integer shift:
    # ========================

    # N even:
    # -------
    N = 10
    x = np.arange(N)
    Fs = 1.
    tau = -1

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        np.concatenate(([x[-1]], x[:-1])))

    # N odd:
    # ------
    N = 11
    x = np.arange(N)
    Fs = 1.
    tau = -1

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        np.concatenate(([x[-1]], x[:-1])))

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
    # as a pure sinusoid, to check that `circular_resample` is performing
    # as expected for non-integer shifts.

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

    # N even:
    # -------
    N = 8
    t, Fs, x = period_N_sine_wrapper(N)
    tau = 0.5 / Fs

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        unity_frequency_sine(t + tau))

    # N odd:
    # ------
    N = 9
    t, Fs, x = period_N_sine_wrapper(N)
    tau = 0.5 / Fs

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        unity_frequency_sine(t + tau))

    # Negative, non-integer shift:
    # ============================

    # N even:
    # -------
    N = 8
    t, Fs, x = period_N_sine_wrapper(N)
    tau = -0.5 / Fs

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        unity_frequency_sine(t + tau))

    # N odd:
    # ------
    N = 9
    t, Fs, x = period_N_sine_wrapper(N)
    tau = -0.5 / Fs

    np.testing.assert_almost_equal(
        rd.signals.sampling.circular_resample(x, Fs, tau),
        unity_frequency_sine(t + tau))

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
    fc = 0.2 * Fs                   # [fc] = [Fs]
    pole = 2                        # [pole] = unitless

    # Spectral density of uncorrelated noise between digital records 1 & 2
    Gnn = 1e-12                     # [Gnn] = [signal]^2 / [Fs]

    # Spectral-estimation parameters
    Nreal_per_ens = 1000            # [Nreal_per_ens] = unitless
    gamma2xy_max = 0.95             # [gamma2xy_max] = unitless

    # Tests:
    # ------
    sig = rd.signals.RandomSignal(Fs, t0, T, fc=fc, pole=pole)
    N = len(sig.x)
    t = sig.t()

    # Generate two digital records that are identical *except* for
    # timebase offset `tau`. Note that `x1[0]` physically occurs at
    # time `t[0]`, while `x2[0]` physically occurs at time `t[0] + tau`.
    x1 = sig.x.copy()
    x2 = rd.signals.sampling.circular_resample(sig.x, Fs, tau)

    # Add uncorrelated noise to generate two distinct signals
    noise_power =  Gnn * (0.5 * Fs)
    noise_amplitude = np.sqrt(noise_power)
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

    # Spectral parameters of broadband signal common to digital records 1 & 2
    fc = 0.2 * Fs                   # [fc] = [Fs]
    pole = 2                        # [pole] = unitless

    # Spectral density of uncorrelated noise between digital records 1 & 2
    Gnn = 1e-12                     # [Gnn] = [signal]^2 / [Fs]

    # Spectral-estimation parameters
    Nreal_per_ens = 1000            # [Nreal_per_ens] = unitless
    gamma2xy_max = 0.95             # [gamma2xy_max] = unitless

    # Tests:
    # ------
    # Common broadband signal
    sig = rd.signals.RandomSignal(Fs, t0, T, fc=fc, pole=pole)
    N = len(sig.x)
    t = sig.t()

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
    noise_power =  Gnn * (0.5 * Fs)
    noise_amplitude = np.sqrt(noise_power)
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
