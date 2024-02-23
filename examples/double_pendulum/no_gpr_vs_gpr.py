from generate_sample import gen_samples
from perform_comparison import generate_rel_error

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from train import build_learner

from scipy import stats

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

def plot_avg_accum_error(time_index, ground_truths, nogpr_trajectories, gpr_trajectories, noise_sd):
    if len(ground_truths) != len(nogpr_trajectories) or len(ground_truths) != len(gpr_trajectories):
        raise ValueError("The lengths of ground_truths, nogpr_trajectories, and gpr_trajectories must be the same.")
        
    if not (len(ground_truths[0]) == len(nogpr_trajectories[0]) == len(gpr_trajectories[0])):
        raise ValueError("The lengths of every trajectory inside ground_truths, nogpr_trajectories, and gpr_trajectories must be the same.")

    plt.figure()
    plt.title(f"Accumulated Error | Noise σ={noise_sd}")
    nogpr_squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
    gpr_squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
    for traj_idx in range(len(ground_truths)):
        nogpr_squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - nogpr_trajectories[traj_idx][0])
        gpr_squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - gpr_trajectories[traj_idx][0])
        for t in range(1, len(ground_truths[0])):
            nogpr_squared_error[traj_idx][t] = nogpr_squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - nogpr_trajectories[traj_idx][t])
            gpr_squared_error[traj_idx][t] = gpr_squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - gpr_trajectories[traj_idx][t])
    mean_nogpr = np.mean(nogpr_squared_error, axis=0)
    mean_gpr = np.mean(gpr_squared_error, axis=0)
    sem_nogpr = stats.sem(nogpr_squared_error, axis=0)
    sem_gpr = stats.sem(gpr_squared_error, axis=0)

    confidence_level = 0.95
    degrees_freedom = len(ground_truths) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error_nogpr = t_critical * sem_nogpr
    margin_of_error_gpr = t_critical * sem_gpr

    plt.plot(time_index, mean_nogpr, color='red', label='No GPR')
    plt.plot(time_index, mean_gpr, color='orange', label='With GPR')
    plt.fill_between(time_index, mean_nogpr - margin_of_error_nogpr, mean_nogpr + margin_of_error_nogpr, 
                        color='red', alpha=0.2, label='95% CI')
    plt.fill_between(time_index, mean_gpr - margin_of_error_gpr, mean_gpr + margin_of_error_gpr, 
                        color='orange', alpha=0.2, label='95% CI')
        
    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated error')
    plt.legend()
    plt.show()

def gen_trajectory_evolution(trained_model_file, testTraj, num_traj):
    with open(trained_model_file, 'rb') as f:
        model = pickle.load(f)
        idx = len(model.learned_weights) - 1
        seed = 101
        # params = model.learned_weights[idx][seed]
        params = model.learned_weights[idx]
        _, _, _, pred_xnext, loss_fun, _, _, _, _ = build_learner(model.nn_hyperparams[0], model.baseline)
        indx_traj = slice(0, num_traj)

        _, _, traj_evol = generate_rel_error((pred_xnext, loss_fun), params, testTraj, indx_traj)

    return traj_evol[101]

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
    print("TESTTRAJSHAPE:", testTraj.shape)
    
    # Paths to trained models, for each percentage of noise, with no/yes GPR Denoising
    trained_models = {
        '0.01': {
            'no': 'pregenerated/trained_models/mlp/0.01gaussiannoise.pkl',
            'yes': 'pregenerated/trained_models/gpmlp/0.01gaussiannoise.pkl'
        },
        '0.05': {
            'no': 'pregenerated/trained_models/mlp/0.05gaussiannoise.pkl',
            'yes': 'pregenerated/trained_models/gpmlp/0.05gaussiannoise.pkl'
        },
        '0.1': {
            'no': 'pregenerated/trained_models/mlp/0.1gaussiannoise.pkl',
            'yes': 'pregenerated/trained_models/gpmlp/0.1gaussiannoise.pkl'
        }
    }

    # Make the comparison for each trained model
    time_axis = jnp.arange(0, num_point_in_traj * actual_dt, actual_dt)
    for noise_level, models in trained_models.items():
        nogpr_results = jnp.array(gen_trajectory_evolution(models['no'], testTraj, num_traj))
        nogpr_results = jnp.transpose(nogpr_results, (1, 0, 2))
        gpr_results = jnp.array(gen_trajectory_evolution(models['yes'], testTraj, num_traj))
        gpr_results = jnp.transpose(gpr_results, (1, 0, 2))

        plot_avg_accum_error(time_axis, testTraj, nogpr_results, gpr_results, noise_level)

main()