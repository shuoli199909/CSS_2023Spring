"""
Implementation of exercise 2 in the course 'Computer Simulations of Sensory Systems' FS2023 in ETH Zurich. 
This exercise is about the simulation of a Vestibular Implant. 
"""

# Author: Shuo Li
# Date: 2023/04/24
# Version: 4

import os
import math
import imageio
import numpy as np
import skinematics as sk
import PySimpleGUI as PSG
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import normalize

"""
Utils for exercise-2.
"""



def sccs_humans():
    """Returns the orientation of the SCCs in humans, relative to Reid's plane.
    Rewrite the provided MATLAB function using Python.
    Provides the orientation of the canals, according to the results of 
    Della Santina et al,  "Orientation of Human Semicircular 
    Canals Measured by Three-Dimensional Multi-planar CT Reconstruction.".
    The orientation of the vectors indicates the direction of stimulation of the corresponding canal.

    Returns
    -------
    canal_left: Orientation of the left canal.
    canal_right: Orientation of the right canal.
    """
    # Left canal
    canal_left = [[-0.32269, -0.03837,  0.94573], 
                  [-0.58930,  0.78839, -0.17655], 
                  [-0.69432, -0.66693, -0.27042]]
    # Right canal
    canal_right = [[ 0.32269, -0.03837, -0.94573], 
                   [ 0.58930,  0.78839,  0.17655], 
                   [ 0.69432, -0.66693,  0.27042]]
    # Normalization
    canal_left = normalize(np.array(canal_left))
    canal_right = normalize(np.array(canal_right))


    return canal_left, canal_right



def alignment(data, vec_ref, vec_g):
    """Align the input data.
    Sensor coords -> Head coords (reference)

    Parameters
    ----------
    data: Vectors in the sensor coordinates.
    vec_ref: Reference vector. 
    vec_g: Gravity vector.

    Returns
    -------
    data_adjusted: Vectors in the head coordinates. Adjusted from the input data.
    """

    # Rotate the sensor to the head coordinates.
    q_rot = np.array([math.sin(math.pi/4), 0, 0])
    # Rotate the gravity into alignment with the reference vector.
    q_short = sk.vector.q_shortest_rotation(vec_ref, vec_g)
    # Quaternion multiplication
    q_total = sk.quat.q_mult(q_rot, q_short)
    # Rotate the input vector into the head coordinates.
    data_adjusted = sk.vector.rotate_vector(data, q_total)


    return data_adjusted



def cal_deflection(stimulus, rate, N):
    """Calculate the stimulus of the semicircular canal and otolith.

    Parameters
    ----------
    stimulus: Semicircular canal stimulus.
    rate: Sampling rate. [Hz]
    N: Number of total samples.

    Returns
    -------
    cupula_deflection: Deflection of the cupula. [rad]
    """

    # SCC dynamics
    # The following implementation is delighted from the tutorial in the lecture.
    # Source:
    # https://nbviewer.org/github/thomas-haslwanter/CSS_ipynb/blob/master/Vestibular_3_SCC_Transduction.ipynb
    T1 = 0.01  #  time-constant for the low-pass filter (sec)
    T2 = 5  #  time-constant for the high-pass filter (sec)

    num = [T1*T2, 0]
    den = [T1*T2, T1+T2, 1]
    # Find the bode-plot, which characterizes the system dynamics
    tf = signal.lti(num, den)  #  SCC transfer function
    # Time axis of the sensory data
    t = np.arange(0, 1./rate*N, 1./rate)
    # Estimate displacement (radians) with calculated SCC_stim_all
    _, cupula_deflection, _ = signal.lsim(system=tf, U=stimulus, T=t)


    return cupula_deflection



def cal_head_orientation(omega_adjusted):
    """Calculate the head orientations during movements.

    Parameters
    ----------
    omega_adjusted: Angular velocities after adjustments.

    Returns
    -------
    head_orientation: The head orientations during movements.
    """

    # Rotate the angular velocities. n_reHead = R('y', -15) * n_measured (anatomical coordinates). 'x/y/z'~'x/-z/y'.
    R_reids = sk.rotmat.R(axis='z', angle=15)
    omega_rot = np.dot(omega_adjusted, R_reids)
    # Calculate the head orientation using quaternions.
    orientation_head = sk.quat.calc_quat(omega_rot, [0,0,0], rate=50, CStype='bf')
    

    return orientation_head



def fun_task_1(data, dir_crt):
    """Task 1 - Simulate the vestibular neural response.
    Contains 2 parts of simulation:
    1. Calculate the maximum cupular displacements.
    2. Calculate the minimum and maximum acceleration along this direction.

    Parameters
    ----------
    data: Data of the fixed sensor.
    dir_crt: Directory of the current folder.

    Returns
    -------
    omega_adjusted: Angular velocities after adjustments.
    """

    print('------Task 1------')

    ## Data decomposition
    acc = data.acc  #  Acceleration
    omega = data.omega  #  Angular velocity
    rate = data.rate  #  Sample rate
    N = data.totalSamples  #  Number of samples

    ## Parameter calculation
    vec_g_hc = np.array([0, 0, -0.9807])  #  Gravity in Zurich. Head coordinates.
    vec_g_approx = np.dot(sk.rotmat.R(axis='x', angle=90), vec_g_hc)
    vec_ref = acc[0, :]  #  Take acc(t=0) as the reference vector.

    ## Displacement of the cupula
    # Adjust the angular velocities.
    omega_adjusted = alignment(data=omega, vec_ref=vec_ref, vec_g=vec_g_approx)
    # Orientation of the SCCs in humans.
    canal_left, canal_right = sccs_humans()
    # Calculate the stimulation using the dot product.
    stimulus = np.dot(omega_adjusted, canal_right[0, :])
    # Calculate deflection
    deflection_cupula = cal_deflection(stimulus, rate, N)
    # Calculate displacement
    radius_canal = 3.2  #  Radius of the SCC
    displacement_cupula = deflection_cupula * radius_canal

    ## Acceleration along the on-direction [0 1 0] in the Head coordinates.
    # Adjust the acceleration.
    acc_adjusted = alignment(data=acc, vec_ref=vec_ref, vec_g=vec_g_approx)
    # Preserve the component in the on-direction [0, 1, 0].
    acc_otolith = acc_adjusted[:, 1]

    ## Write the values in text files.
    # Cupular Displacement
    path_cupular = os.path.join(dir_crt, 'output/CupularDisplacement.txt')
    with open(path_cupular, 'w', encoding='utf-8') as f:
        f.writelines(f'Maximum cupular displacement (positive): \
                     {np.max(displacement_cupula)} mm\n')
        f.writelines(f'Maximum cupular displacement (negative): \
                     {np.min(displacement_cupula)} mm\n')
    print('The maximum cupular displacements have been saved to: ')
    print(path_cupular)
    # Acceleration
    path_acc = os.path.join(dir_crt, 'output/MaxAcceleration.txt')
    with open(path_acc, 'w', encoding='utf-8') as f:
        f.write(f'Maximum acceleration along the direction of the otolith hair cell: \
                {np.max(acc_otolith)} m/s^2\n')
        f.write(f'Minimum acceleration along the direction of the otolith hair cell: \
                {np.min(acc_otolith)} m/s^2\n')
    print('The maximum and minimum acceleration have been saved to: ')
    print(path_acc)
    
    
    return omega_adjusted



def fun_task_2(omega_adjusted, dir_crt):
    """Task 2 - Calculate the "nose-direction" during the movement.
    Visualize the quaternions to describe the head orientation.

    Parameters
    ----------
    omega_adjusted: Angular velocities after adjustments.
    dir_crt: Directory of the current folder.

    Returns
    -------
    orientation_head: Head orientations during the movements.
    orientation_nose: Nose orientations during the movements.
    """

    print('------Task 2------')

    # Head orientation
    orientation_head = cal_head_orientation(omega_adjusted)
    
    # Nose orientation
    orientation_nose = []
    for i in range(orientation_head.shape[0]):
        R_tmp = sk.quat.convert(orientation_head[i, :], to='rotmat')
        # t=0: nose=[1 0 0]
        orientation_nose.append(np.matmul(R_tmp, np.array([1,0,0])))
    orientation_nose = np.array(orientation_nose)

    # Write the values in text files.
    path_nose_dir = os.path.join(dir_crt, 'output/Nose_end.txt')
    with open(path_nose_dir, 'w', encoding='utf-8') as f:
        f.write(f'The orientation if the "Nose"-vector at the end of the walking loop: \
                {orientation_nose[-1]} m/s^2\n')
    print('The nose vector at the end of the walking loop has been saved to:')
    print(path_nose_dir)
    
    # Visualization of the head orientation
    plt.plot(np.linspace(0, 20, orientation_head.shape[0]), orientation_head[:, 1], color='blue')
    plt.plot(np.linspace(0, 20, orientation_head.shape[0]), orientation_head[:, 2], color='green')
    plt.plot(np.linspace(0, 20, orientation_head.shape[0]), orientation_head[:, 3], color='red')
    plt.grid(linestyle='--')
    plt.xlim([0, 20])
    plt.ylim([-0.4, 1.0])
    plt.title('Head Orientation')
    path_img = os.path.join(dir_crt, 'output', 'head_orientation.PNG')
    plt.savefig(path_img)
    print('The visualization result of the head orientation has been saved to:')
    print(path_img)
    

    return orientation_head, orientation_nose



def visualization_nose(orientation_nose, dir_crt):
    """3D visualization of the nose orientation.

    Parameters
    ----------
    orientation_nose: Nose orientations during the movements.
    dir_crt: Directory of the current folder.

    Returns
    -------
    """

    fig = plt.figure()
    # Initialize the 3D coordinates.
    ax = fig.add_subplot(projection='3d')
    # Visualize the nose vectors dynamically.
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y)
    for i in range(len(orientation_nose)):
        # Progress meter
        PSG.one_line_progress_meter(
            'Video frame generating.',
            i+1,
            len(orientation_nose),
            '',
            'Generating frames of the nose orientation...'
        )
        # Clear the current figure.
        plt.cla()
        # 3D surfaces for better visualization
        ax.plot_surface(X, Y, Z=X*0, color='g', alpha=0.2)
        ax.plot_surface(X, Y=X*0, Z=Y, color='y', alpha=0.2)
        ax.plot_surface(X=X*0, Y=Y, Z=X, color='r', alpha=0.2)
        # Plot a 3D arrow (Nose).
        vec_tmp = orientation_nose[i]
        ax.quiver(0, 0, 0, 
                  vec_tmp[0], vec_tmp[1], vec_tmp[2], 
                  arrow_length_ratio=0.2, color='black', normalize=True)
        # Plot settings
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        plt.title('myNose ' + str(i) + '/' + str(len(orientation_nose)))
        path_tmp = os.path.join(dir_crt, 'output', '3D_tmp', str(i)+'.PNG')
        plt.savefig(fname=path_tmp)
    # imageio
    gif_images = []
    for i in range(len(orientation_nose)):
        # Progress meter
        PSG.one_line_progress_meter(
            'GIF generating.',
            i+1,
            len(orientation_nose),
            '',
            'Generating the GIF of the nose orientation...'
        )
        # GIF images
        path_tmp = os.path.join(dir_crt, 'output', '3D_tmp', str(i)+'.PNG')
        gif_images.append(imageio.imread(path_tmp))
    
    # Save the output
    path_save = os.path.join(dir_crt, 'output', 'myNose.gif')
    imageio.mimsave(path_save, gif_images, fps=50)
    print('The GIF which shows the nose orientation has been saved to:')
    print(path_save)


