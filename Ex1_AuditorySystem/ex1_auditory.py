"""
Implementation of exercise 1 in the course 'Computer Simulations of Sensory Systems' FS2023 in ETH Zurich. 
This exercise is about the simulation of Cochlea-Implants. 
"""

# Author: Shuo Li
# Date: 2023/04/01
# Version: 4

import os
import yaml
import numpy as np
import GammaTones as GTS
import PySimpleGUI as sg
from sksound import sounds

'''Implementation of Exercise-1.'''



class Params_audio():
    """Load the pre-defined parameters from a YAML file. Create a class.

    Parameters
    ----------
    path: Path of the YAML file.

    Returns
    -------
    params: A class containing the pre-defined parameters.
    """

    def __init__(self, path_options) -> None:

        # Options.
        self.options = yaml.safe_load(open(path_options))
        # Parameters.
        self.m_electrodes = self.options['ex1']['m_electrodes']  #  Total number of electrodes.
        self.n_channels = self.options['ex1']['n_channels']  #  Number of activated channels.
        self.lo_freq = self.options['ex1']['lo_freq']  #  Minimum frequency.
        self.hi_freq = self.options['ex1']['hi_freq']  #  Maximum frequency.
        self.period_window = self.options['ex1']['period_window']  #  Duration of the sliding window.
        self.step_window = self.options['ex1']['step_window']  #  Duration of the shift step.
        self.type_window = self.options['ex1']['type_window']  #  Sliding window type.



def n_out_m(data_audio, size_window, size_step, Params):
    """The n-out-of-m process.

    Parameters
    ----------
    data_audio: Audio data sequence. One-dimension.
    size_window: Window size. (points)
    size_step: Step size of the moving window. (points)
    params: A class containing the pre-defined parameters.

    Returns
    -------
    amps: Amplitudes of different channels after the n-out-of-m process.
    """

    # Simulate the n_out_of_m process.
    amps = [(data_audio[:, x:x+size_window]**2).sum(axis=1) for x in range(0, data_audio.shape[1], size_step)]
    amps = np.sqrt((np.array(amps)).T)  #  Take the square root. Intensity -> Amplitude.

    # Only select top-n out of m channels.
    for i in range(0, amps.shape[1]):
        idx_zero = (-amps[:, i]).argsort()[Params.n_channels:]
        amps[idx_zero, i] = 0
    
    return amps



def reconstruction(amps, size_step, totalSamples, duration, fcs):
    """Reconstruction process of the output audio.

    Parameters
    ----------
    amps: Amplitude of different channels after the n-out-of-m process.
    size_step: Step size of the moving window (points).
    totalSamples: The desired number of sample points of the output audio. (points)
    duration: The desired duration of the output audio. (sec)
    fcs: Center frequencies. (Hz)

    Returns
    -------
    y_output: The output audio after reconstruction.
    """

    # Audio reconstruction.
    amp_whole = np.repeat(amps, size_step, axis=1)
    amp_whole = amp_whole[:, :totalSamples]  #  Trim the output for alignment.
    # Time sequence (sec).
    t_output = np.linspace(start=0, stop=duration, num=totalSamples)
    y_output = (amp_whole * np.sin(2 * np.pi * np.outer(fcs, t_output))).sum(axis=0)
    # Normalization to rescale the output.
    y_output = y_output/(y_output.max() - y_output.min())

    return y_output



def gui_params(Params):
    """A graphical interface to visualize and change the parameter settings.

    Parameters
    ----------
    Params: A class containing the pre-defined parameters.

    Returns
    -------
    Params: A class containing the parameters after tuning.
    """
    
    # A GUI to show the parameter settings.
    text_1 = sg.Text("These are the pre-defined parameters. You can change them in this window.")
    text_2 = sg.Text("Notes: " 
                     + "If you would like to try out the n-out-of-m strategy used in real cochlear implants, " 
                     + "you should set the number of activated electrodes to be smaller than the total number. " 
                     + "Otherwise, keep these two values to be the same. ")
    text_3 = sg.Text("Parameters: ")

    # Parameters.
    text_m_electrodes_1 = sg.Text("Total number of electrodes:")
    text_m_electrodes_2 = sg.InputText(str(Params.m_electrodes))
    text_n_channels_1 = sg.Text("Number of activated electrodes:")
    text_n_channels_2 = sg.InputText(str(Params.n_channels))
    text_lo_freq_1 = sg.Text("Lowest frequency:")
    text_lo_freq_2 = sg.InputText(str(Params.lo_freq))
    text_hi_freq_1 = sg.Text("Highest frequency:")
    text_hi_freq_2 = sg.InputText(str(Params.hi_freq))
    text_period_window_1 = sg.Text("Duration of the sliding window:")
    text_period_window_2 = sg.InputText(str(Params.period_window))
    text_step_window_1 = sg.Text("Duration of the shift step:")
    text_step_window_2 = sg.InputText(str(Params.step_window))
    text_type_window_1 = sg.Text("Sliding window type:")
    text_type_window_2 = sg.InputText(str(Params.type_window))

    # Buttons.
    button_ok = sg.OK()

    # Layout.
    layout = [[text_1], 
              [text_2], 
              [text_3], 
              [text_m_electrodes_1, text_m_electrodes_2], 
              [text_n_channels_1, text_n_channels_2], 
              [text_lo_freq_1, text_lo_freq_2], 
              [text_hi_freq_1, text_hi_freq_2], 
              [text_period_window_1, text_period_window_2], 
              [text_step_window_1, text_step_window_2], 
              [text_type_window_1, text_type_window_2], 
              [button_ok]]
    
    # Create the window.
    window = sg.Window('Parameter Settings', layout=layout, size=(1500, 300), keep_on_top=True)

    while True:
        event, values = window.read()
        if event in (None, 'OK'):
            # User closed the Window or hit the Cancel button
            break
        print(f'Event: {event}')
        print(str(values))
 
    window.close()

    # Replace the default parameter settings.
    Params.m_electrodes = int(values[0])
    Params.n_channels = int(values[1])
    Params.lo_freq = int(values[2])
    Params.hi_freq = int(values[3])
    Params.period_window = float(values[4])
    Params.step_window = float(values[5])
    Params.type_window = values[6]

    return Params



def main():
    """Main function of exercise 1."""

    # Read data.
    sound = sounds.Sound()
    if len(sound.data.shape) > 1:
        # Average on multiple channels.
        sound.data = np.mean(sound.data, axis=1)

    # Set parameters. Load the parameters from a YAML script.
    path_crt = os.getcwd()
    Params = Params_audio(path_options=os.path.join(path_crt, 'options.yaml'))

    # Create a graphical interface to visualize and change the parameter settings.
    Params = gui_params(Params=Params)

    # Calculate other parameters.
    size_window = round((Params.period_window/sound.duration)*sound.totalSamples)  # window size (points)
    size_step = round((Params.step_window/sound.duration)*sound.totalSamples)  # step size (points)

    # Set up IIR filter banks.
    (forward, feedback, fcs, ERB, B) = GTS.GT_coefficients(
        fs=sound.rate, n_channels=Params.m_electrodes, 
        lo_freq=Params.lo_freq, hi_freq=Params.hi_freq, method=Params.type_window
        )
    
    # Apply filter banks on the whole audio.
    data_filtered = GTS.GT_apply(sound.data, forward, feedback)

    # Simulate the n_out_of_m process.
    amp_n_out_m = n_out_m(
        data_audio=data_filtered, size_window=size_window, 
        size_step=size_step, Params=Params
        )

    # Audio reconstruction.
    y_output = reconstruction(
        amps=amp_n_out_m, size_step=size_step, totalSamples=sound.totalSamples, 
        duration=sound.duration, fcs=fcs
        )

    # Write to output file
    sound_output = sounds.Sound(inData=y_output, inRate=sound.rate)
    sound_output.write_wav(full_out_file=os.path.join(path_crt, 'output.wav'))
    sg.popup("The reconstructed sound has been written to:\n" + os.path.join(path_crt, 'output.wav'))
   


if __name__ == "__main__":
    main()