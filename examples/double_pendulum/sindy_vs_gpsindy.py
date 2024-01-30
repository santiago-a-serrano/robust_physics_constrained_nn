from generate_sample import gen_samples

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from scipy import stats

from sindy_train import SINDy

def print_trajectories(testTraj, n_state, actual_dt, num_traj):
    # Extract the number of points in each trajectory
    num_points = testTraj.shape[1]
    
    # Create time axis based on the number of points and actual_dt
    time_axis = jnp.arange(0, num_points * actual_dt, actual_dt)

    # Plot for each dimension
    for dim in range(n_state):
        plt.figure()
        plt.title(f'Dimension {dim+1} through time for all trajectories')
        for traj in range(num_traj):
            plt.plot(time_axis, testTraj[traj, :, dim], label=f'Trajectory {traj+1}')
        plt.xlabel('Time')
        plt.ylabel(f'Dimension {dim+1} Value')
        plt.legend()
        plt.show()

def plot_avg_accum_error(time_index, ground_truths, sindy_trajectories, gpsindy_trajectories, noise_sd):
    if len(ground_truths) != len(sindy_trajectories) or len(ground_truths) != len(gpsindy_trajectories):
        raise ValueError("The lengths of ground_truths, sindy_trajectories, and gpsindy_trajectories must be the same.")
        
    if not (len(ground_truths[0]) == len(sindy_trajectories[0]) == len(gpsindy_trajectories[0])):
        raise ValueError("The lengths of every trajectory inside ground_truths, sindy_trajectories, and gpsindy_trajectories must be the same.")

    plt.figure()
    plt.title(f"Accumulated Error | Noise σ={noise_sd}")
    sindy_squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
    gpsindy_squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
    for traj_idx in range(len(ground_truths)):
        sindy_squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - sindy_trajectories[traj_idx][0])
        gpsindy_squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - gpsindy_trajectories[traj_idx][0])
        for t in range(1, len(ground_truths[0])):
            sindy_squared_error[traj_idx][t] = sindy_squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - sindy_trajectories[traj_idx][t])
            gpsindy_squared_error[traj_idx][t] = gpsindy_squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - gpsindy_trajectories[traj_idx][t])
    mean_sindy = np.mean(sindy_squared_error, axis=0)
    mean_gpsindy = np.mean(gpsindy_squared_error, axis=0)
    sem_sindy = stats.sem(sindy_squared_error, axis=0)
    sem_gpsindy = stats.sem(gpsindy_squared_error, axis=0)

    confidence_level = 0.95
    degrees_freedom = len(ground_truths) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error_sindy = t_critical * sem_sindy
    margin_of_error_gpsindy = t_critical * sem_gpsindy

    plt.plot(time_index, mean_sindy, color='red', label='SINDy')
    plt.plot(time_index, mean_gpsindy, color='orange', label='GPSINDy')
    plt.fill_between(time_index, mean_sindy - margin_of_error_sindy, mean_sindy + margin_of_error_sindy, 
                        color='red', alpha=0.2, label='95% CI')
    plt.fill_between(time_index, mean_gpsindy - margin_of_error_gpsindy, mean_gpsindy + margin_of_error_gpsindy, 
                        color='orange', alpha=0.2, label='95% CI')
        
    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated error')
    plt.legend()
    plt.show()

def main():
    # Parameters, modify as needed
    actual_dt = 0.01
    num_traj = 10
    num_point_in_traj = 100
    n_state = 4
    maxval_noise = 0.05
    seed = 5

    lowU_test = jnp.asarray([-0.514, -0.514, -0.6, -0.6])
    highU_test = jnp.asarray([0.514, 0.514, 0.6 , 0.6])

    # Generate num_traj random trajectories
    m_rng = jax.random.PRNGKey(seed)
    m_rng, subkey = jax.random.split(m_rng)

    x_init = jax.random.uniform(subkey, shape=(n_state,),
                                    minval=lowU_test, maxval=highU_test)

    m_rng, subkey = jax.random.split(m_rng)
    x_init_lb = x_init - \
        jax.random.uniform(subkey, shape=(n_state,),
                           minval=0, maxval=maxval_noise)
    m_rng, subkey = jax.random.split(m_rng)
    x_init_ub = x_init + \
        jax.random.uniform(subkey, shape=(n_state,),
                           minval=0, maxval=maxval_noise)

    m_rng, testTraj, _ = gen_samples(
        m_rng, actual_dt, num_traj, num_point_in_traj, 1, x_init_lb, x_init_ub, merge_traj=False)
    
    # Get the file to train with
    data_file_path = input("Enter the path to the data file to train SINDY with (ex. DEST_FILE/datatrain_noise1prcnt.pkl): ")
    
    try:
        sindy_results = np.load(f'{data_file_path}_{num_traj}_sindy_cache.npy')
        print("USING SAVED SINDy")
    except:
        print("Generating SINDy...")
        sindy = SINDy(data_file_path, gpsindy=False)
        sindy_results = [sindy.predict(traj[0], num_point_in_traj) for traj in testTraj]
        np.save(f'{data_file_path}_{num_traj}_sindy_cache.npy', sindy_results)

    try:
        gp_sindy_results = np.load(f'{data_file_path}_{num_traj}_gpsindy_cache.npy')
        print("USING SAVED GPSINDy")
    except:
        print("Generating GPSINDy...") 
        manual_hyperparams = bool(int(input("Manually input hyperparameters for Gaussian Process Regression? (1 or 0)")))

        default_sigma_f = 1
        default_sigma_l = 1
        default_sigma_n = 1

        # ALREADY OPTIMIZED HYPERPARAMS:
        # DEST_FILE/datatrain_noise1prcnt.pkl
        # sigma_f 1.1278384923934937
        # sigma_l 36.86850357055664
        # sigma_n -0.020595718175172806
        # DEST_FILE/0.02noise_data.pkl
        # sigma_f 1.2648652791976929
        # sigma_l 38.85000991821289
        # sigma_n -0.038919415324926376
        # DEST_FILE/0.03noise_data.pkl
        # sigma_f 1.261597752571106
        # sigma_l 39.12261199951172
        # sigma_n 0.058368489146232605
        # DEST_FILE/datatrain_noise5prcnt.pkl
        # sigma_f 1.2665953636169434
        # sigma_l 40.35478591918945
        # sigma_n -0.09694703668355942
        # DEST_FILE/datatrain_noise10prcnt.pkl
        # sigma_f 2.7360308170318604
        # sigma_l 37.511592864990234
        # sigma_n 0.19348782300949097
        # DEST_FILE/0.08noise_data.pkl
        # sigma_f 2.903993606567383
        # sigma_l 35.471458435058594
        # sigma_n 0.1528458297252655
        
        if manual_hyperparams:
            default_sigma_f = float(input("Default sigma f: "))
            default_sigma_l = float(input("Default sigma l: "))
            default_sigma_n = float(input("Default sigma n: "))

        gp_sindy = SINDy(data_file_path, gpsindy=True, default_sigma_f=default_sigma_f, 
                                                    default_sigma_l=default_sigma_l, 
                                                    default_sigma_n=default_sigma_n,
                                                    optimize_hyperparams=(not manual_hyperparams))
        gp_sindy_results = [gp_sindy.predict(traj[0], num_point_in_traj) for traj in testTraj]
        np.save(f'{data_file_path}_{num_traj}_gpsindy_cache.npy', gp_sindy_results)


    # Plot
    time_axis = jnp.arange(0, num_point_in_traj * actual_dt, actual_dt)
    noise_sd = input("Enter the noise SD (just for the title of the plot): ")
    plot_avg_accum_error(time_axis, testTraj, sindy_results, gp_sindy_results, noise_sd)

main()