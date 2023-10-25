"""
Implementation of exercise 2 in the course 'Computer Simulations of Sensory Systems' FS2023 in ETH Zurich. 
This exercise is about the simulation of a Vestibular Implant. 
"""

# Author: Shuo Li
# Date: 2023/04/24
# Version: 4

import os
import utils_ex2
from skinematics.sensors.xsens import XSens

'''Implementation of Exercise-2.'''



def main():
    """Main function of Exercise-2."""
    # Load sensor data
    dir_data = os.getcwd()
    data_sensor = XSens(os.path.join(dir_data, 'MovementData', 'Walking_02.txt'))
    # Task-1
    omega_adjusted = utils_ex2.fun_task_1(data=data_sensor, dir_crt=dir_data)
    # Task-2
    orientation_head, orientation_nose = utils_ex2.fun_task_2(omega_adjusted=omega_adjusted, dir_crt=dir_data)
    # Visualization of the nose orientation
    utils_ex2.visualization_nose(orientation_nose=orientation_nose, dir_crt=dir_data)



if __name__ == "__main__":
    main()