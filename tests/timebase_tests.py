from nose import tools
import numpy as np
import random_data as rd


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


def test_TriggerOffset_2signals():
    # Generate two digital records that are identical
    # *except* for timebase offset `tau`
    sig = rd.signals.RandomSignal(Fs, t0, T, fc=fc, pole=pole)
    t = sig.t()
    x1 = sig.x.copy()
    x2 = np.interp(t, t - tau, sig.x)
    N = len(sig.x)

    # Add uncorrelated noise to generate two distinct signals
    noise_power =  Gnn * (0.5 * Fs)
    noise_amplitude = np.sqrt(noise_power)
    x1 += (noise_amplitude * np.random.randn(N))
    x2 += (noise_amplitude * np.random.randn(N))

    # Estimate trigger offset
    trig = rd.timebase.TriggerOffset(
        np.array([x1, x2]),
        Fs=Fs,
        Nreal_per_ens=Nreal_per_ens,
        gamma2xy_max=gamma2xy_max,
        shifts=(np.int(tau * Fs) + shifts))

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
    x2_corrected = np.interp(t - trig.tau, t, x2)

    # The trigger offset between `x1` and `x2_corrected` should be minimal
    trig_corrected = rd.timebase.TriggerOffset(
        np.array([x1, x2_corrected]),
        Fs=Fs,
        Nreal_per_ens=Nreal_per_ens,
        gamma2xy_max=gamma2xy_max,
        shifts=shifts)

    # Trigger offset of corrected signals is at least 200x smaller than
    # original, uncorrected trigger offset
    tools.assert_almost_equal(
        trig_corrected.tau / trig.tau,
        0,
        places=2)

    return


def test_TriggerOffset_4signals():
    # Common broadband signal
    sig = rd.signals.RandomSignal(Fs, t0, T, fc=fc, pole=pole)
    t = sig.t()
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

    # Create additional cross phase between correlation pair
    # `(y1, x2)` by adding a trigger offset `tau` to `x2` and
    # `y2` relative to `x1` and `y1`
    x2 = np.interp(t, t - tau, x2)
    y2 = np.interp(t, t - tau, y2)

    # Finally, add uncorrelated noise
    noise_power =  Gnn * (0.5 * Fs)
    noise_amplitude = np.sqrt(noise_power)
    x1 += (noise_amplitude * np.random.randn(N))
    y1 += (noise_amplitude * np.random.randn(N))
    x2 += (noise_amplitude * np.random.randn(N))
    y2 += (noise_amplitude * np.random.randn(N))

    # Estimate trigger offset
    trig = rd.timebase.TriggerOffset(
        np.array([x1, y1, x2, y2]),
        Fs=Fs,
        Nreal_per_ens=Nreal_per_ens,
        gamma2xy_max=gamma2xy_max,
        shifts=(np.int(tau * Fs) + shifts))

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
    x2_corrected = np.interp(t - trig.tau, t, x2)
    y2_corrected = np.interp(t - trig.tau, t, y2)

    # The trigger offset between 1 and 2 should be minimal
    trig_corrected = rd.timebase.TriggerOffset(
        np.array([x1, y1, x2_corrected, y2_corrected]),
        Fs=Fs,
        Nreal_per_ens=Nreal_per_ens,
        gamma2xy_max=gamma2xy_max,
        shifts=shifts)

    # Trigger offset of corrected signals is at least 200x smaller than
    # original, uncorrected trigger offset
    tools.assert_almost_equal(
        trig_corrected.tau / trig.tau,
        0,
        places=2)

    return
