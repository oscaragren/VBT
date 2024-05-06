import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np


def calculate_velocity(coordinates, delta_time):
    """
    Numerically approximate the velocity of an exercise, and the mean and peak velocity of the concentric phase.

    Parameters:
    coordinates (list): The positional (vertical) information of the exercise
    delta_time (float): The difference in time between two frames.

    Returns:
    list: The approximated velocities of the concentric and (if applicable) eccentric phase
    list: The approixmated velocities of the concentric phase
    float: The peak velocity of the concentric phase
    float: The mean velocitiy of the concentric phase
    """
    all_velocities = [(coordinates[i+1] - coordinates[i]) / delta_time for i in range(len(coordinates)-1)] # Numerical derivative to obtian velocity
    concentric_velocities = [v for v in all_velocities if v >= 0] # Filter the velocities where the velocities is positive (concentric phase)

    peak_velocity = max(concentric_velocities) # Get the peak velocity of concentric phase
    mean_velocity = np.mean(concentric_velocities) # Get the mean velocity of concentric phase

    return all_velocities, concentric_velocities, peak_velocity, mean_velocity

def calculate_rom(data):
    """
    Calculate the Range of Motion of the exercise by subtracting the maximal and minimal value of the positional data.

    Parameters:
    data (list): The positional (vertical) information of an exercise.

    Returns:
    float: The Range of Motion of the exercise
    """
    return np.ptp(data)

def plot_positions_and_velocities(time, positions, velocities, dir_name, exercise, n, model_name):
    """
    Plot the position and velocitiy of an exercise.

    Parameters:
    time (list): The time series of the an exercise
    positions (list): The positional (vertical) information of an exercise
    velocities (list): The velocities of an exercise
    dir_name (str): The name of the directory where the plots are stored
    exercise (str): The type of exercise (squat, bench or deadlift)
    n (int): The repetition of an exercise
    model_name (str): The name of the model used in inference of the video.
    """
    _, axs = plt.subplots(1, 2, figsize=(20, 8)) # Create plot
    
    filtered_position = gaussian_filter1d(positions, sigma=3)
    filtered_velocitiy = gaussian_filter1d(velocities, sigma=3)

    # Position
    axs[0].plot(time, positions, label='Position')
    axs[0].plot(time, filtered_position, linestyle=":", label="Filtered position")
    axs[0].set_title('Position')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Height (m)')

    # Velocity
    axs[1].plot(time[:-1], velocities, label='Velocity')
    axs[1].plot(time[:-1], filtered_velocitiy, linestyle=":", label="Filtered velocity")
    axs[1].set_title('Velocity')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (m/s)')

    plt.tight_layout() 

    plt.savefig(f"{dir_name}/{exercise}_{model_name}_{n}.jpg")
