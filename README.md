Python tools for random data analysis.


Background:
-----------
This module (`random_data`) aims to provide Python tools for
flexible and extensible analysis of random signals.
The motivation and mathematical underpinnings of this module
are largely discussed in Bendat and Piersol's classic text "Random Data", and
inquisitive users are directed there for a thorough discussion
of the methodologies implemented in this module.

The `matplotlib.mlab` submodule offers several useful routines
for spectral density estimation, *but*

    - `psd` and `csd`, which use Welch's average periodogram method
      to estimate the autospectral density and cross-spectral density,
      respectively, are not time-resolved, and

    - `specgram`, which creates a time-resolved estimate of the
      autospectral density (i.e. a "spectrogram"), does not
      average spectral estimates over several realizations, so
      the resulting spectrogram suffers from large amounts
      of random error

In contrast, `random_data` produces time-resolved spectral density estimates
that have been averaged over several realizations, effectively merging
the functionality of `psd`/`csd` and `specgram`. This is largely accomplished
via class definitions built around `matplotlib`'s `psd` and `csd` functions.
Additionally, `random_data`s spectral density classes have robust methods
for visualizing the magnitude, coherence, and phase angle
of spectral density estimates.


Installation:
-------------
Change to the directory you'd like to download the source files to
and retrieve the source files from github by typing

    $ git clone https://github.com/emd/random_data.git

Change into the `random_data` top-level directory by typing

    $ cd random_data

For accounts with root access, install by running

    $ python setup.py install

For accounts without root access (e.g. a standard account on GA's Venus
cluster), install locally by running

    $ python setup.py install --user

To test your installation, run

    $ nosetests tests/

If the tests return "OK", the installation should be working.
