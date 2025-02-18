import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter

# define a function to smooth the trajectory using Savitzky-Golay filter
def smooth_traj_SG(waypoints, window_size, poly_order):
    # Extracting x, y, and theta values from waypoints
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    theta = waypoints[:, 2]

    # Apply the Savitzky-Golay filter to each dimension separately
    smooth_x = savgol_filter(x, window_size, poly_order)
    smooth_y = savgol_filter(y, window_size, poly_order)
    smooth_theta = theta

    # Combine the smoothed values into a new set of waypoints
    smoothed_waypoints = np.column_stack((smooth_x, smooth_y, smooth_theta))
    return smoothed_waypoints

if __name__=='__main__':
    # Load the trajectory waypoints
    trajs = np.load('initialTrajs/initial_trajs_test_N_32_v4_smoothed.npy', allow_pickle=True)
    smoothed_trajs = []
    for traj in trajs:
        smoothed_traj = smooth_traj_SG(traj, min(len(traj), 15), min(len(traj)//2, 3))
        smoothed_trajs.append(smoothed_traj)
    smoothed_trajs = np.array(smoothed_trajs)
    # print the number of vehicles and the time horizon information
    print('The number of vehicles is {}'.format(len(trajs)))
    print('The time horizon is {} units'.format(len(trajs[0])))
    # np.save('initialTrajs/initial_trajs_test_N_32_v4_smoothed_2.npy', smoothed_trajs)
    # waypoints shape: (100, 3) where each row represents [x, y, theta]
    
    # Plot the smoothed trajectories and the original trajectories with different line type and dots in the same figure
    cmap = cm.get_cmap('tab20', len(trajs))
    colors = [cmap(i) for i in range(len(trajs))]
    plt.figure()
    for i in range(len(trajs)):
        if i == 0:
            plt.plot(trajs[i][:,0], trajs[i][:,1], '--', label='original', color = colors[i])
            plt.plot(smoothed_trajs[i][:,0], smoothed_trajs[i][:,1], label='smoothed', color = colors[i])
            # the dot size is 0.1
            plt.scatter(smoothed_trajs[i][:,0], smoothed_trajs[i][:,1], color = colors[i], s=3)
        else:
            plt.plot(trajs[i][:,0], trajs[i][:,1], '--', color = colors[i])
            plt.plot(smoothed_trajs[i][:,0], smoothed_trajs[i][:,1], color = colors[i])
            plt.scatter(smoothed_trajs[i][:,0], smoothed_trajs[i][:,1], color = colors[i],s=3)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.legend()
    plt.show()
