"""
Implementation of exercise 3 in the course 'Computer Simulations of Sensory Systems' FS2023 in ETH Zurich.
This exercise is about the simulation of a Retinal/Visual Implant. 
"""

# Author: Shuo Li
# Date: 2023/05/26
# Version: 4

import os
import math
import yaml
import cv2
import numpy as np
import PySimpleGUI as PSG
import matplotlib.pyplot as plt
from scipy import signal

'''Utils for Exercise-3.'''



class Params_visual():
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
        # Input image.
        self.size_img_default = self.options['ex3']['size_img_default']  #  Image size. (pix).
        # Task-1.
        self.dist_eye_obj = self.options['ex3']['dist_eye_obj']  #  Distance from the eye to the object plane. (m)
        self.ratio_size_pixel = self.options['ex3']['ratio_size_pixel']  #  Conversion factor. (pix/m)
        self.ratio_ecc_rfs = self.options['ex3']['ratio_ecc_rfs']  #  Conversion factor. (arcmin/m)
        self.radius_eye = self.options['ex3']['radius_eye']  #  Radius of the eye. (m)
        self.num_interval = self.options['ex3']['num_interval']  #  Number of divided intervals.
        self.t_ganglion_ON = self.options['ex3']['t_ganglion_ON']  #  Threshold of ON cells.
        self.t_ganglion_OFF = self.options['ex3']['t_ganglion_OFF']  #  Threshold of OFF cells.
        # Task-2.
        self.sigma_gabor = self.options['ex3']['sigma_gabor']  #  Parameters for Gabor filters.
        self.Lambda_gabor = self.options['ex3']['Lambda_gabor']
        self.kernel_size_gabor = self.options['ex3']['kernel_size_gabor']



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
    text_1 = PSG.Text("These are the pre-defined parameters. You can change them in this window.")
    text_2 = PSG.Text("Parameters: ")

    # Parameters for task-1.
    text_task_1 = PSG.Text("Parameters for task-1.")
    text_dist_eye_obj_1 = PSG.Text("Distance from the eye to the object plane:")
    text_dist_eye_obj_2 = PSG.InputText(str(Params.dist_eye_obj))
    text_ratio_size_pixel_1 = PSG.Text("Conversion factor (pix/m):")
    text_ratio_size_pixel_2 = PSG.InputText(str(Params.ratio_size_pixel))
    text_ratio_ecc_rfs_1 = PSG.Text("Conversion factor (arcmin/m):")
    text_ratio_ecc_rfs_2 = PSG.InputText(str(Params.ratio_ecc_rfs))
    text_radius_eye_1 = PSG.Text("Radius of the eye:")
    text_radius_eye_2 = PSG.InputText(str(Params.radius_eye))
    text_num_interval_1 = PSG.Text("Number of divided intervals:")
    text_num_interval_2 = PSG.InputText(str(Params.num_interval))
    text_t_ganglion_ON_1 = PSG.Text("Threshold of ON cells:")
    text_t_ganglion_ON_2 = PSG.InputText(str(Params.t_ganglion_ON))
    text_t_ganglion_OFF_1 = PSG.Text("Threshold of OFF cells:")
    text_t_ganglion_OFF_2 = PSG.InputText(str(Params.t_ganglion_OFF))

    # Parameters for task-2.
    text_task_2 = PSG.Text("Parameters for task-2.")
    text_sigma_gabor_1 = PSG.Text("Sigma:")
    text_sigma_gabor_2 = PSG.InputText(str(Params.sigma_gabor))
    text_Lambda_gabor_1 = PSG.Text("Lambda:")
    text_Lambda_gabor_2 = PSG.InputText(str(Params.Lambda_gabor))
    text_kernel_size_gabor_1 = PSG.Text("Kernel size:")
    text_kernel_size_gabor_2 = PSG.InputText(str(Params.kernel_size_gabor))

    # Buttons.
    button_ok = PSG.OK()

    # Layout.
    layout = [[text_1],
              [text_2],
              # Task-1.
              [text_task_1],
              [text_dist_eye_obj_1, text_dist_eye_obj_2],
              [text_ratio_size_pixel_1, text_ratio_size_pixel_2],
              [text_ratio_ecc_rfs_1, text_ratio_ecc_rfs_2],
              [text_radius_eye_1, text_radius_eye_2],
              [text_num_interval_1, text_num_interval_2],
              [text_t_ganglion_ON_1, text_t_ganglion_ON_2],
              [text_t_ganglion_OFF_1, text_t_ganglion_OFF_2],
              # Task-2.
              [text_task_2],
              [text_sigma_gabor_1, text_sigma_gabor_2],
              [text_Lambda_gabor_1, text_Lambda_gabor_2],
              [text_kernel_size_gabor_1, text_kernel_size_gabor_2],
              [button_ok]]
    
    # Create the window.
    window = PSG.Window('Parameter Settings', layout=layout, keep_on_top=True)

    while True:
        event, values = window.read()
        if event in (None, 'OK'):
            # User closed the Window or hit the Cancel button
            break
        print(f'Event: {event}')
        print(str(values))
 
    window.close()

    # Replace the default parameter settings.
    # Task-1.
    Params.dist_eye_obj = float(values[0])
    Params.ratio_size_pixel = float(values[1])
    Params.ratio_ecc_rfs = float(values[2])
    Params.radius_eye = float(values[3])
    Params.num_interval = int(values[4])
    Params.t_ganglion_ON = float(values[5])
    Params.t_ganglion_OFF = float(values[6])
    # Task-2.
    Params.sigma_gabor = float(values[7])
    Params.Lambda_gabor = float(values[8])
    Params.kernel_size_gabor = int(values[9])


    return Params



def onclick(event):
    """Onclick event.
    Return the mouse movements.

    Parameters
    ----------
    event: Mouse events.

    Returns
    -------
    None.
    """

    coords = [event.xdata, event.ydata]
    # Save the mouse coordinates.
    np.save('coords_tmp', np.array(coords))
    # Close the current figure.
    plt.close()



def coord2interval(coords_fixation, img_input, Params):
    """Find the appropriate intervals.
    This function transforms the input coordinates into different intervals.
    The process is:
    1. Calculate the largest distance from the fixatio point to corners.
    2. Divide the distance into several intervals.
    3. Different intervals correspond to different zones.

    Parameters
    ----------
    coords_fixation: Coordinates of the fixation point.
    img_input: The input image in grayscale.
    Params: A class containing the pre-defined parameters.

    Returns
    -------
    coords_interval: Coordinates of mid-points of different intervals.
    zones: Each pixel is labeled with a number. Same number corresponds to same zone.
    """

    # Get the corner coordinates.
    corners = np.array([
        [0, 0],  #  Upper left.
        [img_input.shape[0], 0],  #  Upper right.
        [0, img_input.shape[1]],  #  Lower left.
        [img_input.shape[0], img_input.shape[1]]  #  Lower right.
    ])

    # Calculate the largest distance from the fixation point to corners.
    dists_corner = np.linalg.norm(coords_fixation-corners, axis=1)
    corner_far = corners[np.argmax(dists_corner)]  #  Farthest corner.

    # Divide the distance into several intervals.
    coords_interval_x = np.linspace(
        start=coords_fixation[0], 
        stop=corner_far[0], 
        num=2 * Params.num_interval + 1
        )
    coords_interval_y = np.linspace(
        start=coords_fixation[1], 
        stop=corner_far[1], 
        num=2 * Params.num_interval + 1
        )

    # Only take the mid-points of the intervals.
    coords_interval_x = coords_interval_x[np.arange(1, 2 * Params.num_interval, 2)]
    coords_interval_y = coords_interval_y[np.arange(1, 2 * Params.num_interval, 2)]

    # 2D coordinates.
    coords_interval = np.concatenate(
        [
        coords_interval_x.reshape([len(coords_interval_x), 1]), 
        coords_interval_y.reshape([len(coords_interval_y), 1])
        ], 
        axis=1
        )

    # Create coordinates of each pixel.
    coords_whole = np.meshgrid(
        np.arange(0, img_input.shape[0]), 
        np.arange(0, img_input.shape[1])
        )
    coords_whole = np.concatenate(
        [coords_whole[0].reshape([np.prod(img_input.shape), 1]), 
         coords_whole[1].reshape([np.prod(img_input.shape), 1])
        ], 
        axis=1
    )

    # Create the zone array.
    zones = np.empty_like(img_input, dtype=int)

    # Calculate the distance from each pixel to the fixation point.
    dists_whole = np.linalg.norm(coords_fixation - coords_whole, axis=1)

    # Length of each interval.
    len_interval = dists_corner.max()/Params.num_interval

    # Assign the zone number to each pixel.
    for i in range(1, Params.num_interval + 1):
        len_tmp = i * len_interval
        idx = (dists_whole >= (len_tmp - len_interval)) & (dists_whole < len_tmp)
        coords_tmp = coords_whole[idx, :]
        zones[coords_tmp[:, 1], coords_tmp[:, 0]] = i
    
    # img_coord ≠ array_coord.

    # Visualization.
    fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
    img_combine = img_input + 0.2*zones  #  Combine the original image with zone numbers.
    img_combine = 1*(img_combine - img_combine.min())/(img_combine.max() - img_combine.min())
    plt.imshow(X=img_combine, cmap='gray')
    plt.title(label='Divide the original image into different zones.')
    plt.scatter(x=coords_interval_x, y=coords_interval_y, c='red')  #  Mid-points.
    plt.plot(
        [coords_fixation[0], corner_far[0]], 
        [coords_fixation[1], corner_far[1]], 
        c='blue'
        )  #  Fixation point -> Farthest corner.
    plt.xlim([0, img_combine.shape[0]])
    plt.ylim([img_combine.shape[1], 0])
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    

    return coords_interval, zones



def get_DoG_filter(img, dist_pix, Params):
    """Get a DoG filter according to requirments.
    This function computes the parameters of the DoG filter.

    Parameters
    ----------
    img: The original image.
    dist_pix: The distance from the fixation point to the mid-point. (pixel)
    Params: A class containing the pre-defined parameters.

    Returns
    -------
    filter_DoG: The returned DoG filter.
    """

    # (pixel) -> (m).
    dist_m = dist_pix / Params.ratio_size_pixel

    # Calculate the angle. (rad)
    angle = math.atan(dist_m/Params.dist_eye_obj)

    # Eccentricity. (m)
    ecc = angle * Params.radius_eye

    # (eccentricity) -> (RFS).
    rfs = ecc * Params.ratio_ecc_rfs
    rfs = rfs * (2 * math.pi) / (360 * 60)  #  (arcmin) -> (rad).

    # (RFS) -> (side length). (eye) -> (object plane).
    len_side = math.tan(rfs) * Params.dist_eye_obj
    len_side = len_side * Params.ratio_size_pixel  #  (m) -> (pixel).

    # Compute the kernel size.
    size_kernel = (2 * int(len_side / 2) + 1)
    size_kernel = np.max([size_kernel, 3])

    # Compute DoG parameters [sigma_1, sigma_2].
    sigma_1 = len_side/8
    sigma_2 = 1.6 * sigma_1

    # Create the DoG filter.
    filter_sigma_1 = cv2.getGaussianKernel(ksize=size_kernel, sigma=sigma_1)
    filter_sigma_1 = np.outer(a=filter_sigma_1, b=filter_sigma_1.T)  #  (1D) -> (2D).
    filter_sigma_2 = cv2.getGaussianKernel(ksize=size_kernel, sigma=sigma_2)
    filter_sigma_2 = np.outer(a=filter_sigma_2, b=filter_sigma_2.T)  #  (1D) -> (2D).
    filter_DoG = filter_sigma_2 - filter_sigma_1


    return filter_DoG



def get_Gabor_filter(theta, Params):
    """Get a Gabor filter according to requirments.
    This function computes the parameters of the Gabor filter.
    Some of the code is based on a demo written by Thomas Haslwanter in gabor_demo.py.

    Parameters
    ----------
    theta: Orientation of the Gabor filter. (deg).
    Params: A class containing the pre-defined parameters.

    Returns
    -------
    filter_Gabor: The returned Gabor filter.
    """
    # Get the parameters
    sigma  = Params.sigma_gabor
    Lambda = Params.Lambda_gabor
    theta = theta * math.pi / 180  #  (deg) -> (rad).
    psi    = 0.5 * math.pi
    kernel_size  = Params.kernel_size_gabor
    
    # make a grid
    xs = np.linspace(-1., 1., kernel_size)
    ys = np.linspace(-1., 1., kernel_size)
    x, y = np.meshgrid(xs, ys)

    x_theta =  x * np.cos(theta) + y * np.sin(theta)
    y_theta = - x * np.sin(theta) + y * np.cos(theta)

    filter_Gabor = np.array(np.exp(-0.5*(x_theta**2+y_theta**2)/sigma**2)*np.cos(2.*np.pi*x_theta/Lambda + psi),dtype=np.float32)
    
    
    return filter_Gabor



def fun_task_1(img_input, Params):
    """Task-1 in Exercise-3.
    Task-1 is about the simulation of the activity in the retinal ganglion cells.
    The whole process is:
    1. Calculate the largest distance from the fixation point to the four corners.
    2. Divide the distance into several intervals and create circular zones.
    3. Determine one DoG filter for each circular zone separately. 
    4. Apply the DoG filters to the input image.

    Parameters
    ----------
    img_input: The input image in grayscale. Numpy array.
    Params: A class containing the pre-defined parameters.

    Returns
    -------
    response_ON: Response reaction of ON ganglion cells.
    response_OFF: Response reaction of OFF ganglion cells.
    """

    # Visualization of the input image.
    fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
    plt.imshow(
        X=cv2.resize(src=img_input, dsize=tuple(Params.size_img_ori) ,interpolation=cv2.INTER_LINEAR),
        cmap='gray'
        )  #  Display in grayscale.
    plt.title(label='The original image.')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Return the fixation coordinates.
    dir_crt = os.getcwd()
    coords_fixation = np.load(os.path.join(dir_crt, 'coords_tmp.npy'), allow_pickle=True)
    # Resize
    coords_fixation[0] = coords_fixation[0]*Params.size_img_default[0]/Params.size_img_ori[0]
    coords_fixation[1] = coords_fixation[1]*Params.size_img_default[1]/Params.size_img_ori[1]

    # Calculate the farthest distance to the corner and divide the distance into several intervals.
    coords_interval, zones = coord2interval(
        coords_fixation=coords_fixation, 
        img_input=img_input, 
        Params=Params
        )
    
    # Apply the appropriate DoG filter to each level.
    img_output = np.empty_like(img_input)
    for i_zone in range(1, Params.num_interval + 1):
        # Calculate the distance between the fixation point and the interval mid-point.
        dist_tmp = np.linalg.norm(coords_fixation - coords_interval[i_zone-1, :])
        # Apply the DoG filter.
        filter_DoG = get_DoG_filter(img=img_input, dist_pix=dist_tmp, Params=Params)
        img_tmp = signal.convolve2d(in1=img_input, in2=filter_DoG, mode='same')
        idx = (zones == i_zone)
        img_output[idx] = img_tmp[idx]
    
    # Simulate the ON-OFF reaction.
    # ON cells.
    response_ON = np.empty_like(img_output)
    response_ON[img_output > Params.t_ganglion_ON] = 1
    response_ON[img_output <= Params.t_ganglion_ON] = 0
    # OFF cells.
    response_OFF = np.empty_like(img_output)
    response_OFF[img_output > Params.t_ganglion_OFF] = 0
    response_OFF[img_output <= Params.t_ganglion_OFF] = 1

    # Resize to the original scale.
    response_ON = cv2.resize(src=response_ON, dsize=tuple(Params.size_img_ori), interpolation=cv2.INTER_LINEAR)
    response_OFF = cv2.resize(src=response_OFF, dsize=tuple(Params.size_img_ori), interpolation=cv2.INTER_LINEAR)

    # Visualization of the ON-OFF reaction.
    fig, ax = plt.subplots(dpi=200, figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(X=response_ON, cmap='gray')  #  Display in grayscale.
    plt.title(label='Responses of ON ganglion cells.')
    plt.subplot(1, 2, 2)
    plt.imshow(X=response_OFF, cmap='gray')
    plt.title(label='Responses of OFF ganglion cells.')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    
    return response_ON, response_OFF



def fun_task_2(img_input, Params):
    """Task-2 in Exercise-3.
    Task-2 is about the simulation of the activities in V1.
    The whole process is:
    1. Calculate Gabor filters of different orientations.
    2. Apply them to the input image separately.
    3. Combine the results and take the average.

    Parameters
    ----------
    img_input: The input image in grayscale. Numpy array.
    Params: A class containing the pre-defined parameters.

    Returns
    -------
    img_output: The output image in grayscale. Numpy array.
    """

    # First, only use Gabor filters which correspond to vertical lines.
    filter_gabor_vertical = get_Gabor_filter(theta=180, Params=Params)
    img_vertical = signal.convolve2d(in1=img_input, in2=filter_gabor_vertical, mode='same')
    img_vertical[img_vertical<0] = 0

    # Then, try different orientations for Gabor filters.
    thetas = np.arange(0, 181, 30)
    img_output = np.empty(shape=(img_input.shape[0], img_input.shape[1], len(thetas)))
    for i_theta in range(len(thetas)):
        filter_gabor_tmp = get_Gabor_filter(theta=thetas[i_theta], Params=Params)
        img_output[:, :, i_theta] = signal.convolve2d(in1=img_input, in2=filter_gabor_tmp, mode='same')
    img_output = np.mean(a=img_output, axis=2)
    img_output[img_output<0] = 0

    # Resize to the original scale.
    img_vertical = cv2.resize(src=img_vertical, dsize=tuple(Params.size_img_ori), interpolation=cv2.INTER_LINEAR)
    img_output = cv2.resize(src=img_output, dsize=tuple(Params.size_img_ori), interpolation=cv2.INTER_LINEAR)

    # Visualization of the output image.
    fig, ax = plt.subplots(dpi=200, figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(X=img_vertical, cmap='gray')  #  Display in grayscale.
    plt.title(label='Only use the Gabor filter with vertical lines.')
    plt.subplot(1, 2, 2)
    plt.imshow(X=img_output, cmap='gray')
    plt.title(label='Use the Gabor filters in different orientations.')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


    return img_output