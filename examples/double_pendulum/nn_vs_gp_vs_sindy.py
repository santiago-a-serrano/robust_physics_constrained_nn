from generate_sample import gen_samples
from perform_comparison import generate_rel_error
from no_gpr_vs_gpr import gen_trajectory_evolution
from sindy_train import SINDy

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from train import build_learner

from scipy import stats

# TODO: Use this code also in the other files, to avoid repeating the plot_avg_accum_error function
# Plot the average accumulated error (and its confidence interval) for the trajectories passed
def plot_avg_accum_error(time_index, ground_truths, trajectories_dict, colors_dict, noise_sd):
    plt.figure()
    plt.title(f"Accumulated Error | Noise σ={noise_sd}")

    # Calculate the accumulated error for all models and all trajectories
    for model_name, trajectories in trajectories_dict.items():
        squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
        for traj_idx in range(len(ground_truths)):
            squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - trajectories[traj_idx][0])
            for t in range(1, len(ground_truths[0])):
                squared_error[traj_idx][t] = squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - trajectories[traj_idx][t])

        # Get the mean (and standard error) of the squared errors (mean of all trajectories)
        mean = np.mean(squared_error, axis=0)
        sem = stats.sem(squared_error, axis=0)

        # Perform calculations needed for plotting the confidence intervals
        confidence_level = 0.95
        degrees_freedom = len(ground_truths) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_of_error = t_critical * sem

        # Plot the mean accumulated errors with their corresponding confidence intervals
        plt.plot(time_index, mean, color=colors_dict[model_name], label=model_name)
        plt.fill_between(time_index, mean - margin_of_error, mean + margin_of_error, color=colors_dict[model_name], alpha=0.2)

    plt.xlabel('Time (s)')
    plt.ylabel("Accumulated error")
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
    
    # Paths to trained models, for each percentage of noise, with no/yes GPR Denoising
    trained_models = {
        '0.01': {
            'MLP': 'pregenerated/trained_models/mlp/0.01gaussiannoise.pkl',
            'GPMLP': 'pregenerated/trained_models/gpmlp/0.01gaussiannoise.pkl',
            'PhysMLP': 'pregenerated/trained_models/physmlp/0.01gaussiannoise.pkl',
            'GPPhysMLP': 'pregenerated/trained_models/gpphysmlp/0.01gaussiannoise.pkl'
        },
        '0.05': {
            'MLP': 'pregenerated/trained_models/mlp/0.05gaussiannoise.pkl',
            'GPMLP': 'pregenerated/trained_models/gpmlp/0.05gaussiannoise.pkl',
            'PhysMLP': 'pregenerated/trained_models/physmlp/0.05gaussiannoise.pkl',
            'GPPhysMLP': 'pregenerated/trained_models/gpphysmlp/0.05gaussiannoise.pkl'
        },
        '0.1': {
            'MLP': 'pregenerated/trained_models/mlp/0.1gaussiannoise.pkl',
            'GPMLP': 'pregenerated/trained_models/gpmlp/0.1gaussiannoise.pkl',
            'PhysMLP': 'pregenerated/trained_models/physmlp/0.1gaussiannoise.pkl',
            'GPPhysMLP': 'pregenerated/trained_models/gpphysmlp/0.1gaussiannoise.pkl'
        }
    }

    colors_dict = {
        'MLP': 'blue',
        'GPMLP': 'cyan',
        'PhysMLP': 'green',
        'GPPhysMLP': 'lime',
        'SINDy': 'red',
        'GPSINDy': 'orange'
    }

    # Optimized hyperparams for training data with varying levels of noise (for GPR Denoising of SINDy and GPSINDy)
    optimized_hyperparams = {
        '0.01': {
            'sigma_f': 1.1278384923934937,
            'sigma_l': 36.86850357055664,
            'sigma_n': -0.020595718175172806
        },
        '0.05': {
            'sigma_f': 1.2665953636169434,
            'sigma_l': 40.35478591918945,
            'sigma_n': -0.09694703668355942
        },
        '0.1': {
            'sigma_f': 2.7360308170318604,
            'sigma_l': 37.511592864990234,
            'sigma_n': 0.19348782300949097
        },
    }

    # Paths with the data to train SINDy and GPSINDy with
    sindy_train_paths = {
        '0.01': 'pregenerated/trajectories/gaussian_noise/0.01.pkl',
        '0.05': 'pregenerated/trajectories/gaussian_noise/0.05.pkl',
        '0.1': 'pregenerated/trajectories/gaussian_noise/0.1.pkl'
    }

    # Make the comparison for each trained model and SINDy/GPSINDy
    all_predicted_trajectories = {}
    for noise_level, models in trained_models.items():
        # for neural network models
        for model_name, model in models.items():
            model_results = jnp.array(gen_trajectory_evolution(model, testTraj, num_traj))
            model_results = jnp.transpose(model_results, (1, 0, 2))
            all_predicted_trajectories[model_name] = model_results
        # for SINDy and GPSINDy
        sindy = SINDy(sindy_train_paths[noise_level], gpsindy=False)
        sindy_results = [sindy.predict(traj[0], num_point_in_traj) for traj in testTraj]
        gp_sindy = SINDy(sindy_train_paths[noise_level], gpsindy=True, 
                         default_sigma_f=optimized_hyperparams[noise_level]['sigma_f'],
                         default_sigma_l=optimized_hyperparams[noise_level]['sigma_l'],
                         default_sigma_n=optimized_hyperparams[noise_level]['sigma_n'],
                         optimize_hyperparams=False)
        gp_sindy_results = [gp_sindy.predict(traj[0], num_point_in_traj) for traj in testTraj]

        all_predicted_trajectories['SINDy'] = sindy_results
        all_predicted_trajectories['GPSINDy'] = gp_sindy_results

        # Plot
        time_axis = jnp.arange(0, num_point_in_traj * actual_dt, actual_dt)
        plot_avg_accum_error(time_axis, testTraj, all_predicted_trajectories, colors_dict, noise_level)

main()