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

# TODO: Hide the 95% CI label
# TODO: Use this code also in the other files, to avoid repeating the plot_avg_accum_error function
def plot_avg_accum_error(time_index, ground_truths, trajectories_dict, colors_dict, noise_sd):
    plt.figure()
    plt.title(f"Accumulated Error | Noise σ={noise_sd}")
    for model_name, trajectories in trajectories_dict.items():
        squared_error = np.zeros((len(ground_truths), len(ground_truths[0])))
        for traj_idx in range(len(ground_truths)):
            squared_error[traj_idx][0] = np.linalg.norm(ground_truths[traj_idx][0] - trajectories[traj_idx][0])
            for t in range(1, len(ground_truths[0])):
                squared_error[traj_idx][t] = squared_error[traj_idx][t-1] + np.linalg.norm(ground_truths[traj_idx][t] - trajectories[traj_idx][t])
        mean = np.mean(squared_error, axis=0)
        sem = stats.sem(squared_error, axis=0)

        confidence_level = 0.95
        degrees_freedom = len(ground_truths) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_of_error = t_critical * sem

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
        '0.02': {
            'MLP': 'pregenerated/trained_models/mlp/0.02advnoise.pkl',
            'GRMLP': 'pregenerated/trained_models/grmlp/0.02advnoise.pkl',
            'PhysMLP': 'pregenerated/trained_models/physmlp/0.02advnoise.pkl',
            'GRPhysMLP': 'pregenerated/trained_models/grphysmlp/0.02advnoise.pkl'
        },
        '0.03': {
            'MLP': 'pregenerated/trained_models/mlp/0.03advnoise.pkl',
            'GRMLP': 'pregenerated/trained_models/grmlp/0.03advnoise.pkl',
            'PhysMLP': 'pregenerated/trained_models/physmlp/0.03advnoise.pkl',
            'GRPhysMLP': 'pregenerated/trained_models/grphysmlp/0.03advnoise.pkl'
        },
        '0.05': {
            'MLP': 'pregenerated/trained_models/mlp/0.05advnoise.pkl',
            'GRMLP': 'pregenerated/trained_models/grmlp/0.05advnoise.pkl',
            'PhysMLP': 'pregenerated/trained_models/physmlp/0.05advnoise.pkl',
            'GRPhysMLP': 'pregenerated/trained_models/grphysmlp/0.05advnoise.pkl'
        }
    }

    colors_dict = {
        'MLP': 'red',
        'GRMLP': 'orange',
        'PhysMLP': 'green',
        'GRPhysMLP': 'lime'
    }


    # Make the comparison for each trained model and SINDy/GPSINDy
    all_predicted_trajectories = {}
    for noise_level, models in trained_models.items():
        # for neural network models
        for model_name, model in models.items():
            model_results = jnp.array(gen_trajectory_evolution(model, testTraj, num_traj))
            model_results = jnp.transpose(model_results, (1, 0, 2))
            all_predicted_trajectories[model_name] = model_results

        # Plot
        time_axis = jnp.arange(0, num_point_in_traj * actual_dt, actual_dt)
        plot_avg_accum_error(time_axis, testTraj, all_predicted_trajectories, colors_dict, noise_level)

main()