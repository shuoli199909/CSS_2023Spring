"""
Implementation of exercise 3 in the course 'Computer Simulations of Sensory Systems' FS2023 in ETH Zurich. 
This exercise is about the simulation of a Retinal/Visual Implant. 
"""

# Author: Shuo Li
# Date: 2023/05/26
# Version: 4

import os
import sys
import cv2
import numpy as np
import PySimpleGUI as PSG
import matplotlib.pyplot as plt
import Ex3_Visual_utils as utils
from skimage.color import rgb2gray

'''Implementation of Exercise-3.'''



def main():
    """Main function of Exercise-3."""
    
    # Load the pre-defined parameters.
    dir_crt = os.getcwd()
    Params = utils.Params_visual(path_options=os.path.join(dir_crt, 'options.yaml'))
    
    # GUI.
    Params = utils.gui_params(Params=Params)

    # Select the input image.
    layout = [[PSG.Text('Select the original image (Browse or type in directly).')], 
              [PSG.InputText(), PSG.FileBrowse('Select Image')],
              [PSG.OK(), PSG.Cancel()]]
    window = PSG.Window('Select the input', layout=layout, keep_on_top=True)

    while True:
        event, values = window.read()
        if event in (None, 'OK'):
            # User hit the OK button.
            break
        elif event in (None, 'Cancel'):
            # User hit the cancel button.
            sys.exit()
        print(f'Event: {event}')
        print(str(values))
 
    window.close()
    img_input = plt.imread(fname=values['Select Image'])
    if len(img_input.shape) == 3:  #  Color image
        img_input = rgb2gray(img_input)
    Params.size_img_ori = np.flip(np.array(img_input.shape))
    #img_input = cv2.resize(src=img_input, dsize=Params.size_img_default)
    img_input = cv2.resize(img_input, dsize=tuple(Params.size_img_default), interpolation=cv2.INTER_LINEAR)
    

    # Task-1.
    # Simulation of the activity in the retinal ganglion cells.
    response_ON, response_OFF = utils.fun_task_1(img_input=img_input, Params=Params)

    # Task-2.
    # Simulation of the activities in V1.
    img_combined = utils.fun_task_2(img_input=img_input, Params=Params)

    # Clear cache.
    os.remove(os.path.join(dir_crt, 'coords_tmp.npy'))

    # Write these outputs to the current folder.
    path_on = os.path.join(dir_crt, 'output', 'response_ON.PNG')
    plt.imsave(fname=path_on, arr=response_ON, cmap='gray')
    path_off = os.path.join(dir_crt, 'output', 'response_OFF.PNG')
    plt.imsave(fname=path_off, arr=response_OFF, cmap='gray')
    path_combined = os.path.join(dir_crt, 'output', 'img_combined.PNG')
    plt.imsave(fname=path_combined, arr=img_combined, cmap='gray')
    print('The response reaction of ON ganglion cells has been save in:')
    print(path_on)
    print('The response reaction of OFF ganglion cells has been save in:')
    print(path_off)
    print('The combined image has been save in:')
    print(path_combined)



if __name__ == "__main__":
    main()