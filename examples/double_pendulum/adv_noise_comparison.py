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

def plot_avg_accum_error(time_index, ground_truths, aprch1_noreg, aprch1_withreg, aprch2_noreg, aprch2_withreg, noise_sd):
    plt.figure()
    plt.title(f"Accumulated Error | Noise σ={noise_sd}")
    aprch1_noreg_squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
    aprch1_withreg_squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
    aprch2_noreg_squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
    aprch2_withreg_squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
    for traj_idx in range(len(ground_truths)):
        aprch1_noreg_squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - aprch1_noreg[traj_idx][0])
        aprch1_withreg_squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - aprch1_withreg[traj_idx][0])
        aprch2_noreg_squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - aprch2_noreg[traj_idx][0])
        aprch2_withreg_squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - aprch2_withreg[traj_idx][0])
        for t in range(1, len(ground_truths[0])):
            aprch1_noreg_squared_error[traj_idx][t] = aprch1_noreg_squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - aprch1_noreg[traj_idx][t])
            aprch1_withreg_squared_error[traj_idx][t] = aprch1_withreg_squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - aprch1_withreg[traj_idx][t])
            aprch2_noreg_squared_error[traj_idx][t] = aprch2_noreg_squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - aprch2_noreg[traj_idx][t])
            aprch2_withreg_squared_error[traj_idx][t] = aprch2_withreg_squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - aprch2_withreg[traj_idx][t])

    # TODO: Continue from here
    mean_aprch1_noreg = np.mean(aprch1_noreg_squared_error, axis=0)
    mean_aprch1_withreg = np.mean(aprch1_withreg_squared_error, axis=0)
    sem_aprch1_noreg = stats.sem(aprch1_noreg_squared_error, axis=0)
    sem_aprch1_withreg = stats.sem(aprch1_withreg_squared_error, axis=0)
    mean_aprch2_noreg = np.mean(aprch2_noreg_squared_error, axis=0)
    mean_aprch2_withreg = np.mean(aprch2_withreg_squared_error, axis=0)
    sem_aprch2_noreg = stats.sem(aprch2_noreg_squared_error, axis=0)
    sem_aprch2_withreg = stats.sem(aprch2_withreg_squared_error, axis=0)

    confidence_level = 0.95
    degrees_freedom = len(ground_truths) - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error_aprch1_noreg = t_critical * sem_aprch1_noreg
    margin_of_error_aprch1_withreg = t_critical * sem_aprch1_withreg
    margin_of_error_aprch2_noreg = t_critical * sem_aprch2_noreg
    margin_of_error_aprch2_withreg = t_critical * sem_aprch2_withreg

    plt.plot(time_index, mean_aprch1_noreg, color='red', label='A1: No GR')
    plt.plot(time_index, mean_aprch1_withreg, color='orange', label='A1: With GR')
    plt.fill_between(time_index, mean_aprch1_noreg - margin_of_error_aprch1_noreg, mean_aprch1_noreg + margin_of_error_aprch1_noreg, 
                        color='red', alpha=0.2, label='95% CI')
    plt.fill_between(time_index, mean_aprch1_withreg - margin_of_error_aprch1_withreg, mean_aprch1_withreg + margin_of_error_aprch1_withreg, 
                        color='orange', alpha=0.2, label='95% CI')
    
    plt.plot(time_index, mean_aprch2_noreg, color='blue', label='A2: No GR')
    plt.plot(time_index, mean_aprch2_withreg, color='cyan', label='A2: With GR')
    plt.fill_between(time_index, mean_aprch2_noreg - margin_of_error_aprch2_noreg, mean_aprch2_noreg + margin_of_error_aprch2_noreg, 
                        color='blue', alpha=0.2, label='95% CI')
    plt.fill_between(time_index, mean_aprch2_withreg - margin_of_error_aprch2_withreg, mean_aprch2_withreg + margin_of_error_aprch2_withreg, 
                        color='cyan', alpha=0.2, label='95% CI')
        
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
        '0.02': {
            'state_noise_no': 'pregenerated/trained_models/mlp/0.02advnoise.pkl',
            'traj_noise_no': 'pregenerated/trained_models/mlp_a2/0.02advnoise.pkl',
            'state_noise_yes': 'pregenerated/trained_models/grmlp/0.02advnoise.pkl',
            'traj_noise_yes': 'pregenerated/trained_models/grmlp_a2/0.02advnoise.pkl'
        },
        '0.03': {
            'state_noise_no': 'pregenerated/trained_models/mlp/0.03advnoise.pkl',
            'traj_noise_no': 'pregenerated/trained_models/mlp_a2/0.03advnoise.pkl',
            'state_noise_yes': 'pregenerated/trained_models/grmlp/0.03advnoise.pkl', 
            'traj_noise_yes': 'pregenerated/trained_models/grmlp_a2/0.03advnoise.pkl'
        },
        '0.05': {
            'state_noise_no': 'pregenerated/trained_models/mlp/0.05advnoise.pkl',
            'traj_noise_no': 'pregenerated/trained_models/mlp_a2/0.05advnoise.pkl',
            'state_noise_yes': 'pregenerated/trained_models/grmlp/0.05advnoise.pkl',
            'traj_noise_yes': 'pregenerated/trained_models/grmlp_a2/0.05advnoise.pkl'
        },
        '0.1': {
            'state_noise_no': 'pregenerated/trained_models/mlp/0.1advnoise.pkl',
            'traj_noise_no': 'pregenerated/trained_models/mlp_a2/0.1advnoise.pkl',
            'state_noise_yes': 'pregenerated/trained_models/grmlp/0.1advnoise.pkl',
            'traj_noise_yes': 'pregenerated/trained_models/grmlp_a2/0.1advnoise.pkl'
        }
    }

    # Make the comparison for each trained model
    time_axis = jnp.arange(0, num_point_in_traj * actual_dt, actual_dt)
    for noise_level, models in trained_models.items():
        state_noise_no = jnp.array(gen_trajectory_evolution(models['state_noise_no'], testTraj, num_traj))
        state_noise_no = jnp.transpose(state_noise_no, (1, 0, 2))
        
        traj_noise_no = jnp.array(gen_trajectory_evolution(models['traj_noise_no'], testTraj, num_traj))
        traj_noise_no = jnp.transpose(traj_noise_no, (1, 0, 2))
        
        state_noise_yes = jnp.array(gen_trajectory_evolution(models['state_noise_yes'], testTraj, num_traj))
        state_noise_yes = jnp.transpose(state_noise_yes, (1, 0, 2))
        
        traj_noise_yes = jnp.array(gen_trajectory_evolution(models['traj_noise_yes'], testTraj, num_traj))
        traj_noise_yes = jnp.transpose(traj_noise_yes, (1, 0, 2))

        plot_avg_accum_error(time_axis, testTraj, state_noise_no, state_noise_yes, traj_noise_no, traj_noise_yes, noise_level)

main()