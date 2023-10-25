"""
Python-port of the clever alorithms from Slaney (1993), which were first
implemented in Matlab by Nick Clarke (2007).
The original ideas have been published in the Apple TR #35, "An Efficient
Implementation of the Patterson-Holdsworth Cochlear Filter Bank." (You can
find this article under "PattersonsEar.pdf" on the WWW).
All formulas are from there, with exeption of the modification of the 
center-frequency calculation.
"""

# author: Thomas Haslwanter
# date:   Mar-2023

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def set_parameters(low_freq: float, method: str = "moore") -> tuple:
    """Sets the parameters for GT_coefficients (below)

    Parameters
    ----------
    low_freq : lower cutoff-frequency, to stop some errors [Hz]
    method : source of chosen parameters. Has to be one of the following
             * greenwood
             * lyon (== stanley)
             * moore (== glasberg) [DEFAULT]
             * wierddsam

    Returns
    -------
    low_freq : verified lower frequency cutoff
    EarQ :
    minBW :
    order :
    """

    #  stop errors when a very low fequency is used
    low_freq = max(low_freq, 75)

    # Based on the "method", set the analysis parameters
    if (method == "lyon") or (method == "stanley"):
        # Lyon + Stanley Parameters (1988)
        EarQ = 8
        minBW = 125
        order = 2
    elif method == "greenwood":
        # Greenwood Parameters (1990) as (nearly) in DSAM
        EarQ = 7.23824
        minBW = 22.8509
        order = 1
    elif (method == "moore") or (method == "glasberg"):
        # Glasberg and Moore Parameters (1990)
        EarQ = 9.26449
        minBW = 24.7
        order = 1
    elif method == "wierddsam":
        EarQ = 9.26
        minBW = 15.719
        order = 1
    else:
        print('Invalid "method" - unp.sing "moore"')
        EarQ = 9.26449
        minBW = 24.7
        order = 1

    return (low_freq, EarQ, minBW, order)


def GT_coefficients(
    fs: float, n_channels: int, lo_freq: float, hi_freq: float, method: str
):
    """Computes the filter coefficients for a bank of Gammatone filters

    These filters were defined by Patterson and Holdworth for simulating the
    cochlea. The results are returned as arrays of filter coefficients. Each row
    of the filter arrays (forward and feedback) can be passed to
    `scipy.signal.lfilter` .

    Parameters
    ----------
    fs : sampling frequency [Hz]
    n_channels : number of Channels
    lo_freq : lower frequency limit
    hi_freq : upper frequency limit
    method : method for finding the parameters

    Returns
    -------
    forward : "b"-coefficients for the linear filter
    feedback : "a"-coefficients for the linear filter
    cf : center frequencies
    ERB : Equivalent Rectangular Bandwidth
    B : Gammatone filter parameter in Roy Patterson's ear model
    """

    fs = float(fs)
    (lo_freq, EarQ, minBW, order) = set_parameters(lo_freq, method)
    T = 1 / fs

    # to make sure that the subsequent calculations are in float
    lo_freq = float(lo_freq)
    hi_freq = float(hi_freq)

    ERBlo = ((lo_freq / EarQ) ** order + minBW ** order) ** (1 / order)
    ERBhi = ((hi_freq / EarQ) ** order + minBW ** order) ** (1 / order)
    overlap = (ERBhi / ERBlo) ** (1.0 / (n_channels - 1))
    ERB = np.array([ERBlo * (overlap ** channel) for channel in range(n_channels)])

    cf = EarQ * ((ERB ** order - minBW ** order) ** (1 / order))
    pi = np.pi
    B = 1.019 * 2 * pi * ERB  # in rad here. Note: some models require B in Hz (NC)

    #    a = (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) - np.sqrt(3 - 2**(3./2))*np.sin(2*cf*pi*T)))
    #    b = (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) + np.sqrt(3 - 2**(3./2))*np.sin(2*cf*pi*T)))
    #    c = (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) - np.sqrt(3 + 2**(3./2))*np.sin(2*cf*pi*T)))
    #    d = (-2*np.exp(4*1j*cf*pi*T)*T +2*np.exp(-(B*T) + 2*1j*cf*pi*T) * T * (np.cos(2*cf*pi*T) + np.sqrt(3 + 2**(3./2))*np.sin(2*cf*pi*T)))
    #    e = (-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cf*pi*T) +2*(1 + np.exp(4*1j*cf*pi*T))/np.exp(B*T))**4

    gain = abs(
        (
            -2 * np.exp(4 * 1j * cf * pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cf * pi * T)
            * T
            * (
                np.cos(2 * cf * pi * T)
                - np.sqrt(3 - 2 ** (3.0 / 2)) * np.sin(2 * cf * pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cf * pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cf * pi * T)
            * T
            * (
                np.cos(2 * cf * pi * T)
                + np.sqrt(3 - 2 ** (3.0 / 2)) * np.sin(2 * cf * pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cf * pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cf * pi * T)
            * T
            * (
                np.cos(2 * cf * pi * T)
                - np.sqrt(3 + 2 ** (3.0 / 2)) * np.sin(2 * cf * pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cf * pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cf * pi * T)
            * T
            * (
                np.cos(2 * cf * pi * T)
                + np.sqrt(3 + 2 ** (3.0 / 2)) * np.sin(2 * cf * pi * T)
            )
        )
        / (
            -2 / np.exp(2 * B * T)
            - 2 * np.exp(4 * 1j * cf * pi * T)
            + 2 * (1 + np.exp(4 * 1j * cf * pi * T)) / np.exp(B * T)
        )
        ** 4
    )

    feedback = np.zeros((len(cf), 9))
    forward = np.zeros((len(cf), 5))

    forward[:, 0] = T ** 4 / gain
    forward[:, 1] = -4 * T ** 4 * np.cos(2 * cf * pi * T) / np.exp(B * T) / gain
    forward[:, 2] = 6 * T ** 4 * np.cos(4 * cf * pi * T) / np.exp(2 * B * T) / gain
    forward[:, 3] = -4 * T ** 4 * np.cos(6 * cf * pi * T) / np.exp(3 * B * T) / gain
    forward[:, 4] = T ** 4 * np.cos(8 * cf * pi * T) / np.exp(4 * B * T) / gain

    feedback[:, 0] = np.ones(len(cf))
    feedback[:, 1] = -8 * np.cos(2 * cf * pi * T) / np.exp(B * T)
    feedback[:, 2] = 4 * (4 + 3 * np.cos(4 * cf * pi * T)) / np.exp(2 * B * T)
    feedback[:, 3] = (
        -8 * (6 * np.cos(2 * cf * pi * T) + np.cos(6 * cf * pi * T)) / np.exp(3 * B * T)
    )
    feedback[:, 4] = (
        2
        * (18 + 16 * np.cos(4 * cf * pi * T) + np.cos(8 * cf * pi * T))
        / np.exp(4 * B * T)
    )
    feedback[:, 5] = (
        -8 * (6 * np.cos(2 * cf * pi * T) + np.cos(6 * cf * pi * T)) / np.exp(5 * B * T)
    )
    feedback[:, 6] = 4 * (4 + 3 * np.cos(4 * cf * pi * T)) / np.exp(6 * B * T)
    feedback[:, 7] = -8 * np.cos(2 * cf * pi * T) / np.exp(7 * B * T)
    feedback[:, 8] = np.exp(-8 * B * T)

    return (forward, feedback, cf, ERB, B)


def GT_apply(x: np.ndarray, forward: np.ndarray, feedback: np.ndarray):
    """This function filters the waveform x with the array of filters
    specified by the forward and feedback parameters. Each row
    of the forward and feedback parameters are passed on to
    `scipy.signal.lfilter` .

    Parameters
    ----------
    x : incoming sound
    forward : "b"-coefficients for the linear filter
    feedback : "a"-coefficients for the linear filter

    Returns
    -------

    """

    # Allocate the memory
    (rows, cols) = np.shape(feedback)
    y = np.zeros([rows, len(x)])

    # Filter the signal
    for ii in range(rows):
        y[ii, :] = signal.lfilter(forward[ii, :], feedback[ii, :], x)

    return y


def show_basilarmembrane_movement(
    stimulus: np.ndarray,
    rate: float,
    fcs: np.ndarray,
    freqs_to_label: list,
    ax: plt.Axes,
) -> None:
    """This is a simple plotting routine to mimic the basilar-membrane-movement
    plot types seen frequently in the "Journal Of The Acoustical Society Of
    America" (JASA) articles among others, as well as software such as AIM and
    AMS.  This allows data from each channel to be viewed as stackedline graphs.

    Parameters
    ----------
    sound : movements of selected spots on the basilar membrane
    rate : sample rate [Hz]
    fcs : center-frequencies
    freqs_to_label : list of integers, which of the many frequencies should be
                     labelled on the y-axis
    ax : axis for the plot
    """

    # plot the different traces above each other, i.e. shift each stimulus up by
    # its row-number
    shifts = np.zeros(np.shape(stimulus))
    for n in range(stimulus.shape[0]):
        shifts[n, :] = n
    stim_plot = stimulus / np.max(abs(stimulus)) + shifts

    # Set the time axis
    time = (np.arange(stimulus.shape[1]) + 1) * 1.0e3 / rate

    # Plot the data
    ax.plot(time, stim_plot.transpose(), "k")

    # Format the plot
    if len(freqs_to_label) > 0:
        ax.set_yticks(freqs_to_label, np.round(fcs[freqs_to_label]))
    else:
        ax.set_yticks([0], "")

    ax.set_ylim(-1, stimulus.shape[0])
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Center Frequency [Hz]")


def main():
    """Test function, with a click-train as input"""

    rate = 16e3  # sampling rate [Hz]
    x = np.zeros(int(rate * 25e-3))  # create a 25ms input
    x[[1, 100, 200, 300]] = 1  # make a click train

    # And now for the real thing: make the filterbank ...
    (forward, feedback, fcs, ERB, B) = GT_coefficients(rate, 50, 200, 3000, "moore")
    print(f'The shape of "forward" is here {forward.shape}')
    print(f'The shape of "feedback" is here {feedback.shape}')

    # ... and filter into individual channels.
    y = GT_apply(x, forward, feedback)

    # Show the plots
    fig, axs = plt.subplots(1, 2)

    # Show all frequencies, and label a selection of centre frequencies
    show_basilarmembrane_movement(y, rate, fcs, [0, 9, 19, 29, 39, 49], axs[0])

    # For better visibility, plot selected center-frequencies in a second plot.
    # Dont plot the centre frequencies on the ordinate.
    show_basilarmembrane_movement(y[[0, 9, 19, 29, 39, 49], :], rate, [], [], axs[1])
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()