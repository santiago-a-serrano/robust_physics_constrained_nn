import numpy as np
import pickle

import matplotlib.pyplot as plt

import time
import argparse

import math

from physics_constrained_nn.utils import SampleLog, LearningLog, HyperParamsNN

from sindy_train import SINDy

from train import build_learner
from generate_sample import gen_samples

import jax
import jax.numpy as jnp

from tqdm.auto import tqdm

from scipy.io import loadmat

from sympy import symbols, sympify



def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=1), axis=1)
    return (cumsum[..., N:] - cumsum[..., :-N]) / float(N)


def combine_list(mList):
    max_size = np.max(np.array([len(elem) for elem in mList]))
    m_new_list = list()
    for elem in mList:
        if len(elem) >= max_size:
            m_new_list.append(elem)
            continue
        elem.extend([elem[-1] for i in range(max_size-len(elem))])
        m_new_list.append(elem)
    return m_new_list


def generate_loss(loss_evol_1traj, window=1):
    """ Generate the data to plot the loss evolution of a fixed trajectory and for its different seed
    """
    # Load the training and testing data total loss function
    loss_tr = np.array(combine_list([dict_val['total_loss_train']
                       for seed, dict_val in sorted(loss_evol_1traj.items())]))
    loss_tr = running_mean(loss_tr, window)
    loss_te = np.array(combine_list([dict_val['total_loss_test']
                       for seed, dict_val in sorted(loss_evol_1traj.items())]))
    loss_te = running_mean(loss_te, window)

    # Load the specific data on the rollout and constraints
    spec_data_tr = np.array(combine_list(
        [dict_val['rollout_err_train'] for seed, dict_val in sorted(loss_evol_1traj.items())]))
    spec_data_te = np.array(combine_list(
        [dict_val['rollout_err_test'] for seed, dict_val in sorted(loss_evol_1traj.items())]))
    # coloc_data_tr = np.array([ dict_val['coloc_err_train'] for seed, dict_val in sorted(loss_evol_1traj.items()) ])
    coloc_data_te = np.array(combine_list(
        [dict_val['coloc_err_test'] for seed, dict_val in sorted(loss_evol_1traj.items())]))

    # Compute the mean and standard deviation of the total loss
    total_loss_tr_mean, total_loss_tr_std = np.mean(
        loss_tr, axis=0), np.std(loss_tr, axis=0)
    total_loss_tr_max, total_loss_tr_min = np.max(
        loss_tr, axis=0), np.min(loss_tr, axis=0)
    total_loss_te_mean, total_loss_te_std = np.mean(
        loss_te, axis=0), np.std(loss_te, axis=0)
    total_loss_te_max, total_loss_te_min = np.max(
        loss_te, axis=0), np.min(loss_te, axis=0)

    # Compute the mean square loss without the penalization and l2 and constraints
    ms_error_tr_roll = running_mean(
        np.mean(spec_data_tr[:, :, :, 1], axis=2), window)
    ms_error_te_roll = running_mean(
        np.mean(spec_data_te[:, :, :, 1], axis=2), window)

    ms_error_tr_mean, ms_error_tr_std = np.mean(
        ms_error_tr_roll, axis=0), np.std(ms_error_tr_roll, axis=0)
    ms_error_tr_max, ms_error_tr_min = np.max(
        ms_error_tr_roll, axis=0), np.min(ms_error_tr_roll, axis=0)
    ms_error_te_mean, ms_error_te_std = np.mean(
        ms_error_te_roll, axis=0), np.std(ms_error_te_roll, axis=0)
    ms_error_te_max, ms_error_te_min = np.max(
        ms_error_te_roll, axis=0), np.min(ms_error_te_roll, axis=0)

    coloc_error_te_roll = running_mean(coloc_data_te[:, :, 0], window)
    coloc_error_te_roll_mean, coloc_error_te_roll_std = np.mean(
        coloc_error_te_roll, axis=0), np.std(coloc_error_te_roll, axis=0)
    coloc_error_te_roll_max, coloc_error_te_roll_min = np.max(
        coloc_error_te_roll, axis=0), np.min(coloc_error_te_roll, axis=0)

    # Compute the value of the constraints without the penalization term
    # constr_error_tr_roll = running_mean(np.mean(spec_data_tr[:,:,:,2], axis=2) + np.mean(spec_data_tr[:,:,:,3], axis=2) +  coloc_data_te[:,:,1] + coloc_data_te[:,:,2], window)
    # constr_error_te_roll = running_mean(np.mean(spec_data_te[:,:,:,2], axis=2) +  np.mean(spec_data_te[:,:,:,3], axis=2) +  coloc_data_te[:,:,1] + coloc_data_te[:,:,2], window)
    constr_error_tr_roll = running_mean(np.mean(spec_data_tr[:, :, :, 2], axis=2) + np.mean(
        spec_data_tr[:, :, :, 3], axis=2) + coloc_data_te[:, :, 1] + coloc_data_te[:, :, 2], window)
    constr_error_te_roll = running_mean(np.mean(spec_data_te[:, :, :, 2], axis=2) + np.mean(
        spec_data_te[:, :, :, 3], axis=2) + coloc_data_te[:, :, 1] + coloc_data_te[:, :, 2], window)
    constr_error_tr_mean, constr_error_tr_std = np.mean(
        constr_error_tr_roll, axis=0), np.std(constr_error_tr_roll, axis=0)
    constr_error_tr_max, constr_error_tr_min = np.max(
        constr_error_tr_roll, axis=0), np.min(constr_error_tr_roll, axis=0)
    constr_error_te_mean, constr_error_te_std = np.mean(
        constr_error_te_roll, axis=0), np.std(constr_error_te_roll, axis=0)
    constr_error_te_max, constr_error_te_min = np.max(
        constr_error_te_roll, axis=0), np.min(constr_error_te_roll, axis=0)

    return (total_loss_tr_mean, total_loss_tr_mean-total_loss_tr_std,  total_loss_tr_mean+total_loss_tr_std), \
        (total_loss_te_mean, total_loss_te_mean-total_loss_te_std, total_loss_te_mean+total_loss_te_std),\
        (ms_error_tr_mean, np.maximum(ms_error_tr_min, ms_error_tr_mean-ms_error_tr_std), np.minimum(ms_error_tr_max, ms_error_tr_mean+ms_error_tr_std)),\
        (ms_error_te_mean, np.maximum(ms_error_te_min, ms_error_te_mean-ms_error_te_std), np.minimum(ms_error_te_max, ms_error_te_mean+ms_error_te_std)),\
        (constr_error_tr_mean, np.maximum(constr_error_tr_min, constr_error_tr_mean-constr_error_tr_std), np.minimum(constr_error_tr_max, constr_error_tr_mean+constr_error_tr_std)),\
        (constr_error_te_mean, np.maximum(constr_error_te_min, constr_error_te_mean-constr_error_te_std), np.minimum(constr_error_te_max, constr_error_te_mean+constr_error_te_std)),\
        (coloc_error_te_roll_mean, np.maximum(coloc_error_te_roll_min, coloc_error_te_roll_mean-coloc_error_te_roll_std),
         np.minimum(coloc_error_te_roll_max, coloc_error_te_roll_mean+coloc_error_te_roll_std))
    # return (total_loss_tr_mean, np.maximum(total_loss_tr_min, total_loss_tr_mean-total_loss_tr_std),np.minimum(total_loss_tr_max, total_loss_tr_mean+total_loss_tr_std)), \
    # 		(total_loss_te_mean, np.maximum(total_loss_te_min, total_loss_te_mean-total_loss_te_std),np.minimum(total_loss_te_max, total_loss_te_mean+total_loss_te_std)),\
    # 		(ms_error_tr_mean, np.maximum(ms_error_tr_min,ms_error_tr_mean-ms_error_tr_std), np.minimum(ms_error_tr_max,ms_error_tr_mean+ms_error_tr_std) ),\
    # 		(ms_error_te_mean, np.maximum(ms_error_te_min,ms_error_te_mean-ms_error_te_std), np.minimum(ms_error_te_max,ms_error_te_mean+ms_error_te_std) ),\
    # 		(constr_error_tr_mean, np.maximum(constr_error_tr_min,constr_error_tr_mean-constr_error_tr_std), np.minimum(constr_error_tr_max,constr_error_tr_mean+constr_error_tr_std)),\
    # 		(constr_error_te_mean, np.maximum(constr_error_te_min,constr_error_te_mean-constr_error_te_std), np.minimum(constr_error_te_max,constr_error_te_mean+constr_error_te_std))

# xtrue contains multiple trajectories
def generate_rel_error(util_fun, params, xtrue, indx_traj=0):
    """ Generate the relative error and geometric mean of the error for some given trajectories of the pendulum

            Probably need to check that the trajectories aren't too different for 
            the standard deviation plot to not be too wide
    """
    # Extract the functions to compute the loss and predict next state
    pred_xnext, loss_fun = util_fun

    # Geometric mean function
    def expanding_gmean_log(s):
        return np.transpose(np.exp(np.transpose(np.log(s).cumsum(axis=0)) / (np.arange(s.shape[0])+1)))

    # A list to save the time evolution of the error per each initialization
    res_value = [list() for i in range(1, xtrue.shape[1])]
    trajectory_evolution = {seed: [xtrue[indx_traj, 0, :]] for seed in params}
    for seed, params_val in sorted(params.items()):
        # The initial point are all initial point in the given trajectories
        init_value = xtrue[:, 0, :]
        print("init_value", init_value)
        # Then we estimate the future time step
        for i, l_res_value in tqdm(zip(range(1, xtrue.shape[1]), res_value), total=len(res_value), leave=False):
            init_value, _, _, _, _, _ = pred_xnext(params_val, init_value)
            # the parameters -> print("params_val", params_val)
            trajectory_evolution[seed].append(init_value[indx_traj, :])
            # Compute the relative error
            curr_relative_error = np.linalg.norm(init_value-xtrue[:, i, :], axis=1)/(
                np.linalg.norm(xtrue[:, i, :], axis=1) + np.linalg.norm(init_value, axis=1))
            l_res_value.extend(curr_relative_error)

    # Make to an array the relative error as a function of initialization
    res_value = np.array(res_value)
    res_value = expanding_gmean_log(res_value)

    # Compute the mean value and standard deviation
    mean_err = np.mean(res_value, axis=1)
    # tqdm.write('{}'.format(mean_err))
    std_err = np.std(res_value, axis=1)
    # Compute the maximum and minim value for proper confidence bound
    max_err_value = np.max(res_value, axis=1)
    min_err_value = np.min(res_value, axis=1)
    maximum_val = np.minimum(max_err_value, mean_err+std_err)
    minimum_val = np.maximum(min_err_value, mean_err-std_err)
    return (mean_err, minimum_val, maximum_val), \
        (mean_err[-1], minimum_val[-1], maximum_val[-1]),\
        trajectory_evolution


def load_files(data_to_show):
    """ Load a series of file to be used in plotting the rewards and the 
    """
    # The list to save the results of the log data
    m_learning_log = []
    # The list to save the functions for predicting the next state and computing the loss function
    m_util_fun = []
    for f_name in data_to_show:
        # Load the file
        m_file = open(f_name, 'rb')
        # Load the Log data
        m_llog = pickle.load(m_file)
        m_file.close()
        # Append the data in the list of logs to render
        m_learning_log.append(m_llog)
        # Parse the file to obtain the function to predict the NN -> Just one element should be enough
        _, _, _, pred_xnext, loss_fun, _, _, _, _ = build_learner(
            m_llog.nn_hyperparams[0], baseline=m_llog.baseline)
        m_util_fun.append((pred_xnext, loss_fun))
        print(m_llog.nn_hyperparams[0])
        print(m_llog.sampleLog)
    return m_learning_log, m_util_fun

from sympy import lambdify

import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, sympify, lambdify

import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, sympify, lambdify

def get_sindypi_result(starting_point, equations_str, dt=0.01, steps=100):
    # Initialize the derivatives list
    z1, z2, z3, z4 = symbols('z1 z2 z3 z4')
    
    # Parse the equations from the input string
    lines = equations_str.split('\n')
    dz = [sympify(lines[i].strip()) for i in range(1, len(lines), 2) if lines[i].strip()]
    
    # Ensure we have 4 equations
    if len(dz) != 4:
        raise ValueError("There must be exactly 4 equations, one for each derivative dz1, dz2, dz3, and dz4.")
    
    # Convert symbolic expressions to numerical functions
    dz_funcs = [lambdify((z1, z2, z3, z4), expr) for expr in dz]
    
    # Define a function that returns the derivatives
    def odes(t, y):
        return [func(*y) for func in dz_funcs]
    
    # Set up the time span and initial conditions
    t_span = (0, dt * steps)
    t_eval = np.linspace(t_span[0], t_span[1], steps)
    
    # Solve the ODE using Runge-Kutta 45
    solution = solve_ivp(odes, t_span, starting_point, method='RK45', t_eval=t_eval)
    
    return solution.y.T

# TODO: Do they count as squared errors? It's the norm of the errors
def plot_squared_errors(time_index, ground_truth, trajectories, colors, labels):
    # Just the squared errors:
    plt.figure()
    plt.title("Squared Errors")
    for traj_idx, trajectory in enumerate(trajectories):
        if trajectory is None:
            continue
        squared_error = [None] * len(ground_truth)
        for i in range(len(ground_truth)):
            squared_error[i] = np.linalg.norm(ground_truth[i] - trajectory[i])
        plt.plot(time_index, squared_error, 
                        color=colors[traj_idx], linewidth=linewidth, label=labels[traj_idx])
        
    plt.xlabel('Time (s)')
    plt.ylabel('Squared error')
    plt.legend()
    plt.show()
    

    # The accumulated squared errors:
    plt.figure()
    plt.title("Accumulated Squared Errors")
    for traj_idx, trajectory in enumerate(trajectories):
        if trajectory is None:
            continue
        squared_error = [None] * len(ground_truth)
        squared_error[0] = np.linalg.norm(ground_truth[0] - trajectory[0])
        for i in range(1, len(ground_truth)):
            squared_error[i] = np.linalg.norm(ground_truth[i] - trajectory[i]) + squared_error[i-1]
        plt.plot(time_index, squared_error, 
                        color=colors[traj_idx], linewidth=linewidth, label=labels[traj_idx])
        
    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated squared error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import time
    import argparse

    # example_command ='python generate_sample.py --cfg dataset_gen.yaml --output_file data/xxxx --time_step 0.01 --n_rollout 5'
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdirs', nargs='+', required=True)
    parser.add_argument('--legends', nargs='+', required=True)
    parser.add_argument('--colors', nargs='+', required=True)
    parser.add_argument('--window', '-w', type=int, default=1)
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--seed', type=int, default=701)
    parser.add_argument('--evaluate_on_test', action='store_true')
    parser.add_argument('--num_traj', type=int, default=100)
    parser.add_argument('--num_point_in_traj', type=int, default=500)
    parser.add_argument('--show_constraints', action='store_true')
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--indx_traj_test', type=int, default=0)
    parser.add_argument('--indx_traj', type=int, default=0)
    parser.add_argument('--train_sindy_path', type=str, default='')
    parser.add_argument('--sindypi_eqs', type=str, default='')
    parser.add_argument('--use_cached_sindys', type=str, default='')
    parser.add_argument('--gpsindy', type=bool, default=False)
    args = parser.parse_args()

    # Load the data
    print("ARGS.LOGDIRS ARE THE FOLLOWING", args.logdirs)
    m_logs, m_pred_loss = load_files(args.logdirs)
    actual_dt = m_logs[0].nn_hyperparams[0].actual_dt
    n_state = m_logs[0].sampleLog.n_state

    ##########################################################################################################################
    # Should define some parameters for the plot here
    alpha_std = 0.4
    alpha_std_test = 0.2
    linewidth = 1.5
    markerstyle = "*"
    markersize = 8
    linestyle_test = {'linestyle': '--', 'dashes': (10, 5)}
    # Generate the figure for the loss function evolution
    m_figs_loss = list()
    for i in range(len(m_logs[0].sampleLog.disable_substep[1])):
        ncols = 2 if args.show_constraints else 1
        nrows = 2 if args.show_constraints else 1
        m_fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows),
                                  sharex=True, sharey=False)
        m_figs_loss.append((m_fig, axs))
        # Set the title

    # Generate the plots for the loss function evolution on all the data
    for (log_data, legend, color) in tqdm(zip(m_logs, args.legends, args.colors), total=len(m_logs)):
        loss_evol_data = {i: val for i,
                          val in log_data.loss_evol.items() if len(val) > 0}
        for traj_id, n_train, (m_fig, axs) in zip(loss_evol_data, log_data.sampleLog.disable_substep[1], m_figs_loss):
            tl_tr, tl_te, ml_tr, ml_te, ctr_tr, ctr_te, coloc_err = generate_loss(
                loss_evol_data[traj_id], args.window)
            temp_nnparams = log_data.nn_hyperparams[0]
            high_freq_record_rg = int(
                temp_nnparams.freq_accuracy[0]*temp_nnparams.num_gradient_iterations)
            high_freq_val = temp_nnparams.freq_accuracy[1]
            low_freq_val = temp_nnparams.freq_accuracy[2]
            update_freq = np.array([(i % high_freq_val) == 0 if i <= high_freq_record_rg else ((i % low_freq_val) == 0 if i < temp_nnparams.num_gradient_iterations-1 else True)
                                    for i in range(temp_nnparams.num_gradient_iterations)])
            gradient_step = np.array(
                [i for i in range(temp_nnparams.num_gradient_iterations)])[update_freq]
            # print(axs)
            main_axs = axs.ravel()[0] if args.show_constraints else axs
            # main_axs.plot(gradient_step[:tl_tr[0].shape[0]], tl_tr[0], color=color, linewidth=linewidth, label=legend)
            # main_axs.fill_between(gradient_step[:tl_tr[0].shape[0]], tl_tr[1], tl_tr[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
            main_axs.plot(gradient_step[:tl_te[0].shape[0]], tl_te[0], color='dark' +
                          color, linewidth=linewidth, label=legend, **linestyle_test)
            main_axs.fill_between(gradient_step[:tl_te[0].shape[0]], tl_te[1],
                                  tl_te[2], linewidth=linewidth, facecolor=color, alpha=alpha_std_test)
            main_axs.set_yscale('symlog')
            main_axs.set_xlabel(r'$\mathrm{Time \ steps}$')
            main_axs.set_ylabel(
                r'$\mathrm{Total \ loss}$ ($N_{\mathrm{train}}=' + str(n_train)+'$)')
            main_axs.grid()
            main_axs.legend(loc='best')
            if args.show_constraints:
                # Print the mean squared error without the l2 and constraint
                if np.sum(ml_tr[0]) > 1e-8:
                    # axs.ravel()[1].plot(gradient_step[:ml_tr[0].shape[0]], ml_tr[0], color=color, linewidth=linewidth, label=legend)
                    # axs.ravel()[1].fill_between(gradient_step[:ml_tr[1].shape[0]], ml_tr[1], ml_tr[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
                    axs.ravel()[1].plot(gradient_step[:ml_te[0].shape[0]], ml_te[0],
                                        color='dark'+color, linewidth=linewidth, label=legend, **linestyle_test)
                    axs.ravel()[1].fill_between(gradient_step[:ml_te[1].shape[0]], ml_te[1],
                                                ml_te[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
                    axs.ravel()[1].set_yscale('log')
                    axs.ravel()[1].set_xlabel(r'$\mathrm{Time \ steps}$')
                    axs.ravel()[1].set_ylabel(
                        r'$\mathrm{Mean \ squared \ loss}$ ($N_{\mathrm{train}}=' + str(n_train)+'$)')
                    axs.ravel()[1].grid()
                    axs.ravel()[1].legend(loc='best')
                # Plot the constraints
                if np.sum(ctr_tr[0]) > 1e-8:
                    # axs.ravel()[2].plot(gradient_step[:ctr_tr[0].shape[0]], ctr_tr[0], color=color, linewidth=linewidth, label=legend)
                    # axs.ravel()[2].fill_between(gradient_step[:ctr_tr[1].shape[0]], ctr_tr[1], ctr_tr[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
                    axs.ravel()[2].plot(gradient_step[:ctr_te[0].shape[0]], ctr_te[0],
                                        color='dark'+color, linewidth=linewidth, label=legend, **linestyle_test)
                    axs.ravel()[2].fill_between(gradient_step[:ctr_te[1].shape[0]], ctr_te[1],
                                                ctr_te[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
                    axs.ravel()[2].set_yscale('log')
                    axs.ravel()[2].set_xlabel(r'$\mathrm{Time \ steps}$')
                    axs.ravel()[2].set_ylabel(
                        r'$\mathrm{Constraints \ loss}$ ($N_{\mathrm{train}}=' + str(n_train)+'$)')
                    axs.ravel()[2].grid()
                    axs.ravel()[2].legend(loc='best')
                # Plot the colocation
                if np.sum(coloc_err[0]) > 1e-8:
                    axs.ravel()[3].plot(gradient_step[:coloc_err[0].shape[0]],
                                        coloc_err[0], color=color, linewidth=linewidth, label=legend)
                    axs.ravel()[3].fill_between(gradient_step[:coloc_err[1].shape[0]], coloc_err[1],
                                                coloc_err[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
                    axs.ravel()[3].set_yscale('log')
                    axs.ravel()[3].set_xlabel(r'$\mathrm{Time \ steps}$')
                    axs.ravel()[3].set_ylabel(
                        r'$\mathrm{Colocation \ Err.}$ ($N_{\mathrm{train}}=' + str(n_train)+'$)')
                    axs.ravel()[3].grid()
                    axs.ravel()[3].legend(loc='best')

    # # Save this figure if required
    import tikzplotlib
    from pathlib import Path
    dir_save = str(Path(args.logdirs[0]).parent)
    for n_train, (m_fig, axs) in zip(log_data.sampleLog.disable_substep[1], m_figs_loss):
        tikzplotlib.clean_figure(fig=m_fig)
        tikzplotlib.save(dir_save+'/loss_{}.tex'.format(n_train), figure=m_fig)
        m_fig.savefig(dir_save+'/loss_{}.png'.format(n_train), dpi=300)
    ##########################################################################################################################

    #########################################################################################################################
    m_rng = jax.random.PRNGKey(args.seed)
    m_rng, subkey = jax.random.split(m_rng)
    # Check if the test should be evaluated on the training set
    if args.evaluate_on_test:
        x_init = jax.random.uniform(subkey, shape=(n_state,),
                                    minval=m_logs[0].sampleLog.lowU_test, maxval=m_logs[0].sampleLog.highU_test)
    else:
        x_init = jax.random.uniform(subkey, shape=(n_state,),
                                    minval=m_logs[0].sampleLog.lowU_train, maxval=m_logs[0].sampleLog.highU_train)
    m_rng, subkey = jax.random.split(m_rng)
    x_init_lb = x_init - \
        jax.random.uniform(subkey, shape=(n_state,),
                           minval=0, maxval=args.noise)
    m_rng, subkey = jax.random.split(m_rng)
    x_init_ub = x_init + \
        jax.random.uniform(subkey, shape=(n_state,),
                           minval=0, maxval=args.noise)
    # Define a rule for generating trajectories for evaluation the realative error
    # testTraj contains num_traj trajectories
    m_rng, testTraj, _ = gen_samples(
        m_rng, actual_dt, args.num_traj, args.num_point_in_traj, 1, x_init_lb, x_init_ub, merge_traj=False)
    time_index = [actual_dt * i for i in range(1, args.num_point_in_traj)]
    # TODO: I need the results of this function for our seed when it's working normally.

    # Generate the plots showing the relative error
    list_axes_rel_err = list()
    for i in range(len(m_logs[0].sampleLog.disable_substep[1])):
        fig_rel_err = plt.figure()
        ax_rel_err = plt.gca()
        ax_rel_err.set_xlabel(r'$\mathrm{Time \ (seconds)}$')
        ax_rel_err.set_yscale('log')
        ax_rel_err.grid()
        list_axes_rel_err.append((ax_rel_err, fig_rel_err))

    figure_gm_rerr = plt.figure()
    ax_gm_rerr = plt.gca()
    ax_gm_rerr.set_xlabel(r'$\mathrm{Number \ of \ training \ trajectories }$')
    ax_gm_rerr.set_ylabel(
        r'$\mathrm{Geometric \ mean \ of \ relative \ error }$')
    ax_gm_rerr.set_yscale('log')
    ax_gm_rerr.grid()

    mStateEvol = dict()
    for pred_loss_fun, log_data, legend, color in tqdm(zip(m_pred_loss, m_logs, args.legends, args.colors), total=len(m_pred_loss)):
        training_list = list()
        list_gm_err_mean, list_gm_err_min, list_gm_err_max = list(), list(), list()
        curr_learned_params = {
            i: val for i, val in log_data.learned_weights.items() if len(val) > 0}
        counterVal = 0
        for traj_id, n_train, (ax_rel_err, _) in tqdm(zip(curr_learned_params, log_data.sampleLog.disable_substep[1], list_axes_rel_err), total=len(list_axes_rel_err), leave=False):
            rel_err, gm_rel_err, trajPred = generate_rel_error(
                pred_loss_fun, curr_learned_params[traj_id], testTraj, args.indx_traj_test)
            if counterVal == args.indx_traj:
                mStateEvol[legend] = trajPred
            list_gm_err_mean.append(float(gm_rel_err[0]))
            list_gm_err_min.append(float(gm_rel_err[1]))
            list_gm_err_max.append(float(gm_rel_err[2]))
            training_list.append(n_train)
            # label=r'$N_{\mathrm{traj}} = '+ str(n_train) + '$'
            ax_rel_err.plot(
                time_index, rel_err[0], color=color, linewidth=linewidth, label=legend)
            ax_rel_err.fill_between(
                time_index, rel_err[1], rel_err[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
            ax_rel_err.set_ylabel(
                r'$\mathrm{Relative \ error}$ ($N_{\mathrm{train}}=' + str(n_train)+'$)')
            ax_rel_err.legend(loc='best')
            tqdm.write('[N_train = {}, Legend = {}]\t : Geometric mean [Mean | Mean-Std | Mean+Std] -> {:.4f} | {:.4f} | {:.4f}'.format(
                n_train, legend, float(gm_rel_err[0]), float(gm_rel_err[1]), float(gm_rel_err[2])))
        ax_gm_rerr.plot(training_list, list_gm_err_mean, color=color, linewidth=linewidth,
                        marker=markerstyle, markersize=markersize, label=legend)
        ax_gm_rerr.fill_between(training_list, list_gm_err_min, list_gm_err_max,
                                linewidth=linewidth, facecolor=color, alpha=alpha_std)
    ax_gm_rerr.legend(loc='best')

    # for pred_loss_fun, log_data, legend, color in tqdm(zip(m_pred_loss, m_logs, args.legends, args.colors), total=len(m_pred_loss)):
    # 	training_list = list()
    # 	list_gm_err_mean, list_gm_err_min, list_gm_err_max  = list(), list(), list()
    # 	for traj_id, n_train, ax_rel_err in tqdm(zip(log_data.learned_weights, log_data.sampleLog.xTrainExtra[0],list_axes_rel_err), total=len(list_axes_rel_err), leave=False):
    # 		rel_err, gm_rel_err = generate_rel_error(pred_loss_fun, log_data.learned_weights[traj_id], testTraj)
    # 		list_gm_err_mean.append(float(gm_rel_err[0]))
    # 		list_gm_err_min.append(float(gm_rel_err[1]))
    # 		list_gm_err_max.append(float(gm_rel_err[2]))
    # 		training_list.append(n_train)
    # 		ax_rel_err.plot(time_index, rel_err[0], color=color, linewidth=linewidth) # label=r'$N_{\mathrm{traj}} = '+ str(n_train) + '$'
    # 		ax_rel_err.fill_between(time_index, rel_err[1], rel_err[2], linewidth=linewidth, facecolor=color, alpha=alpha_std, label=legend)
    # 		ax_rel_err.set_ylabel(r'$\mathrm{Relative \ error}$ ($N_{\mathrm{train}}='+ str(n_train)+'$)')
    # 		ax_rel_err.legend(loc='best')
    # 		tqdm.write('[N_train = {}, Legend = {}]\t : Geometric mean [Mean | Mean-Std | Mean+Std] -> {:.4f} | {:.4f} | {:.4f}'.format(n_train, legend, float(gm_rel_err[0]), float(gm_rel_err[1]), float(gm_rel_err[2])))
    # 	ax_gm_rerr.plot(training_list, list_gm_err_mean, color=color, linewidth=linewidth)
    # 	ax_gm_rerr.fill_between(training_list, list_gm_err_min, list_gm_err_max, linewidth=linewidth, facecolor=color, alpha=alpha_std, label=legend)

    ##########################################################################################################################
    # Plot the state evolution to show the accuracy
    # Save the remaining figures
    import tikzplotlib
    from pathlib import Path
    for n_train, (ax_rel_err, m_fig) in zip(log_data.sampleLog.disable_substep[1], list_axes_rel_err):
        tikzplotlib.clean_figure(fig=m_fig)
        tikzplotlib.save(
            dir_save+'/relerr_{}.tex'.format(n_train), figure=m_fig)
        m_fig.savefig(dir_save+'/relerr_{}.svg'.format(n_train), dpi=300)
    tikzplotlib.clean_figure(fig=figure_gm_rerr)
    tikzplotlib.save(dir_save+'/geomrelerr.tex', figure=figure_gm_rerr)
    figure_gm_rerr.savefig(dir_save+'/geomrelerr.png', dpi=300)

    # Plot the state evolution to show the accuracy
    # Plot the trajectories
    state_name = [r'$\Theta_1$', r'$\Theta_2$', r'$\Omega_1$', r'$\Omega_2$']
    time_index = [actual_dt * i for i in range(args.num_point_in_traj)]
    true_state_evol = testTraj[args.indx_traj_test]
    m_seed = next(iter(mStateEvol[next(iter(mStateEvol))]))
    print("STARTING POINT: ", true_state_evol[0])
    sindy_result = None
    sindy_color = 'grey'
    gpsindy_result = None
    gpsindy_color = 'black'
    sindy_pi_result = None
    sindy_pi_color = 'purple'
    # JUST USE CACHED SINDYS: --use_cached_sindys string
    # GENERATE THE CACHED SINDYS: --train_sindy_path string
    if args.train_sindy_path != '':
            try:
                sindy_result = SINDy(args.train_sindy_path).predict(true_state_evol[0], 100)
                np.save(f'{args.train_sindy_path}_sindy_cache.npy', sindy_result)
            except:
                pass
    if args.gpsindy:
            try:
                gpsindy_result = SINDy(args.train_sindy_path, gpsindy=True).predict(true_state_evol[0], 100)
                np.save(f'{args.train_sindy_path}_gpsindy_cache.npy', gpsindy_result)
            except:
                pass
    if args.sindypi_eqs != '':
            try:
                with open(args.sindypi_eqs, 'r') as file:
                    eqs_string = file.read()
                    sindy_pi_result = get_sindypi_result(true_state_evol[0], eqs_string)
            except IOError:
                print("File not accessible")
            
    if args.use_cached_sindys != '':
        try:
            sindy_result = np.load(f'{args.use_cached_sindys}_sindy_cache.npy')
        except:
            pass
        try:
            gpsindy_result = np.load(f'{args.use_cached_sindys}_gpsindy_cache.npy')
        except:
            pass
    for i in range(len(state_name)):
        plt.figure()
        plt.plot(time_index, true_state_evol[:, i], color='green',
                 linewidth=linewidth, linestyle='dashed', label='True state')
        for legend, color in zip(args.legends, args.colors):
            plt.plot(time_index, np.array(mStateEvol[legend][m_seed])[
                     :, i], color=color, linewidth=linewidth, label=legend)
        # If available, plot the SINDy simulation too
        if sindy_result is not None:
            try:
                plt.plot(time_index, sindy_result[:, i], 
                        color=sindy_color, linewidth=linewidth, label='SINDy')
            except:
                pass
        if gpsindy_result is not None:
            try:
                plt.plot(time_index, gpsindy_result[:, i],
                        color=gpsindy_color, linewidth=linewidth, label='GPSINDy')
            except:
                pass
        if sindy_pi_result is not None:
            try:
                plt.plot(time_index, sindy_pi_result[:, i],
                        color=sindy_pi_color, linewidth=linewidth, label='SINDy-PI')
            except:
                pass
        plt.xlabel('Time (s)')
        plt.ylabel(state_name[i])
        plt.legend(loc='best')
        plt.grid()
        tikzplotlib.clean_figure()
        tikzplotlib.save(dir_save+'/state_{}.tex'.format(i))
        plt.savefig(dir_save+'/state_{}.png'.format(i),
                    dpi=300, transparent=True)
        print(m_seed)
    plt.show()

    # Plot the squared errors of the argument and benchmark trajectories
    trajectories = []
    colors = []
    legends = []
    for legend, color in zip(args.legends, args.colors):
            trajectories.append(np.array(mStateEvol[legend][m_seed]))
            colors.append(color)
            legends.append(legend)

    trajectories.append(sindy_result)
    colors.append(sindy_color)
    legends.append('SINDy')
    trajectories.append(gpsindy_result)
    colors.append(gpsindy_color)
    legends.append('GPSINDy')
    trajectories.append(sindy_pi_result)
    colors.append(sindy_pi_color)
    legends.append('SINDy-PI')

                    
    plot_squared_errors(time_index, true_state_evol, trajectories, colors, legends)


