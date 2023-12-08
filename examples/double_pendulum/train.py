import yaml
from collections import namedtuple
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from generate_sample import pendulum_known_terms, custom_constraints
from physics_constrained_nn.phyconstrainednets import build_learner_with_sideinfo
from physics_constrained_nn.utils import SampleLog, HyperParamsNN, LearningLog
from physics_constrained_nn.utils import _INITIALIZER_MAPPING, _ACTIVATION_FN, _OPTIMIZER_FN
from physics_constrained_nn.noise_generator import add_gaussian_noise
import physics_constrained_nn.gaussian_process_regressor as gpr

import optax

from tqdm import tqdm


def build_params(m_params, output_size=None, input_index=None):
    """ Build the parameters of the neural network from a dictionary
            :param m_params: A dictionary specifying the number of layers and inputs of the neural network
            :param output_size : Provide the size of the output layer of the neural network
            :param input_index : Provide a mapping of the neural network input to the full state represnetation
    """
    # Some sanity check
    assert (output_size is None and input_index is None) or (
        input_index is not None and output_size is not None)

    # Store the resulting neural network
    nn_params = dict()

    # Check if the inputs indexes are given -> If yes this is not the nn for the apriori enclosure
    if input_index is not None:
        nn_params['input_index'] = input_index

    # Check of the size of the output layer is given -> If yes thus us not the nn for the apriori enclosure
    if output_size is not None:
        nn_params['output_sizes'] = (*m_params['output_sizes'], output_size)
    else:
        nn_params['output_sizes'] = m_params['output_sizes']

    # Activation function
    nn_params['activation'] = _ACTIVATION_FN[m_params['activation']]

    # Initialization of the biais values
    b_init_dict = m_params['b_init']
    nn_params['b_init'] = _INITIALIZER_MAPPING[b_init_dict['initializer']](
        **b_init_dict['params'])

    # Initialization of the weight values
    w_init_dict = m_params['w_init']
    nn_params['w_init'] = _INITIALIZER_MAPPING[w_init_dict['initializer']](
        **w_init_dict['params'])

    return nn_params


def build_learner(paramsNN, baseline='rk4', constraint_noise=0):
    """ Given a structure like HyperParamsNN, this function generates the functions to compute the loss,
            to update the weight and to predict the next state
            :param paramsNN : HyperParamsNN structure proving the parameters to build the NN
    """

    # The optimizer for gradient descent over the loss function -> See yaml for more details
    # Optimizer initial learning rate
    lr = paramsNN.optimizer['learning_rate_init']
    # Optimizer end learning rate
    lr_end = paramsNN.optimizer['learning_rate_end']

    # Customize the gradient descent algorithm
    chain_list = [_OPTIMIZER_FN[paramsNN.optimizer['name']]
                  (**paramsNN.optimizer.get('params', {}))]

    # Add weight decay if enable
    decay_weight = paramsNN.optimizer.get('weight_decay', 0.0)
    if decay_weight > 0.0:
        chain_list.append(optax.add_decayed_weights(decay_weight))

    # Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
    m_schedule = optax.linear_schedule(-lr, -
                                       lr_end, paramsNN.num_gradient_iterations)
    chain_list.append(optax.scale_by_schedule(m_schedule))

    # Add gradient clipping if enable
    grad_clip_coeff = paramsNN.optimizer.get('grad_clip', 0.0)
    if grad_clip_coeff > 0.0:
        chain_list.append(optax.adaptive_grad_clip(clipping=grad_clip_coeff))

    # Build the solver finally
    opt = optax.chain(*chain_list)

    # The random number generator
    rng_gen = jax.random.PRNGKey(seed=paramsNN.seed_init)

    # Build nn_params using the input in the yaml file
    dict_params = paramsNN.nn_params.copy()

    # Extract the type of side information
    type_sideinfo = dict_params.pop('type_sideinfo', None)
    assert type_sideinfo is not None, 'The type of side information is not specified'
    ###############################################################

    # Extract the apriori enclosure neural network
    apriori_net = dict_params.pop('apriori_encl', None)
    assert apriori_net is not None, 'The params for the apriori nnn are not specified'
    apriori_net = baseline if (
        baseline == 'base' or baseline == 'rk4') else build_params(apriori_net)
    tqdm.write("{}\n".format(apriori_net))
    ###############################################################

    #################################################################
    # This is specific to the type of environemnt
    # Build the nn depending on the type of side information
    if type_sideinfo == 0:

        # Specify if we train usin constraints
        train_with_constraints = False
        assert len(dict_params) == 1, 'There should be only one remaining key'

        # Build the neural network parameter
        nn_params = {'vector_field': build_params(dict_params['vector_field'],
                                                  input_index=jnp.array(
                                                      [i for i in range(paramsNN.n_state+paramsNN.n_control)]),
                                                  output_size=paramsNN.n_state)}

        # Build the learner main functions
        params, pred_xnext, (loss_fun, loss_fun_constr), (update, update_lagrange) = \
            build_learner_with_sideinfo(rng_gen, opt, paramsNN.model_name, paramsNN.actual_dt,
                                        paramsNN.n_state, paramsNN.n_control,  nn_params, apriori_net, known_dynamics=None,
                                        constraints_dynamics=None, pen_l2=paramsNN.pen_l2, pen_constr=paramsNN.pen_constr,
                                        batch_size=paramsNN.batch_size, train_with_constraints=train_with_constraints)

    ################################################################
    elif type_sideinfo == 1 or type_sideinfo == 2:
        train_with_constraints = False if type_sideinfo == 1 else True
        # Build the neural network parameter
        nn_params = {'f1': build_params(dict_params['vector_field'], input_index=jnp.array([0, 1, 3]), output_size=1),
                     'f2': build_params(dict_params['vector_field'], input_index=jnp.array([0, 1, 2]), output_size=1)}
        params, pred_xnext, (loss_fun, loss_fun_constr), (update, update_lagrange) = build_learner_with_sideinfo(rng_gen, opt, paramsNN.model_name, paramsNN.actual_dt,
                                                                                                                 paramsNN.n_state, paramsNN.n_control,
                                                                                                                 nn_params, apriori_net, known_dynamics=jax.vmap(
                                                                                                                     pendulum_known_terms),
                                                                                                                 constraints_dynamics=custom_constraints, pen_l2=paramsNN.pen_l2,
                                                                                                                 pen_constr=paramsNN.pen_constr, batch_size=paramsNN.batch_size, train_with_constraints=train_with_constraints,
                                                                                                                 constraint_noise=constraint_noise)
    else:
        raise ("Not implemented yet !")

    # Jit the functions for the learner
    pred_xnext = jax.jit(pred_xnext)
    loss_fun = jax.jit(loss_fun)
    update = jax.jit(update)
    update_lagrange = jax.jit(update_lagrange)
    loss_fun_constr = jax.jit(loss_fun_constr)
    return rng_gen, opt, params, pred_xnext, loss_fun, update, update_lagrange, loss_fun_constr, train_with_constraints


def load_config_file(path_config_file, extra_args={}):
    """ Load the yaml configuration file giving the training/testing information and the neural network params
            :param path_config_file : Path to the adequate yaml file
    """
    yml_file = open(path_config_file).read()
    m_config = yaml.load(yml_file, yaml.SafeLoader)
    m_config = {**m_config, **extra_args}
    print('################# Configuration file #################')
    print(m_config)
    print('######################################################')

    # File containing the training and testing samples
    data_set_file = m_config['train_data_file']
    mFile = open(data_set_file, 'rb')
    mSampleLog = pickle.load(mFile)
    mFile.close()

    model_name = m_config['model_name']
    out_file = m_config['output_file']

    type_baseline = m_config.get('baseline', 'rk4')
    weight_noise = m_config.get('weight_noise', 0.1)
    bias_noise = m_config.get('bias_noise', 0.1)
    constraint_noise = m_config.get('constraint_noise', 0.1)
    perform_gp_denoising = m_config.get('perform_gp_denoising', False)
    optimize_hyperparams = m_config.get('optimize_hyperparams', True)
    big_denoising = m_config.get('big_denoising', True)
    default_sigma_n = m_config.get('default_sigma_n', 0.1)

    opt_info = m_config['optimizer']

    seed_list = m_config['seed']

    batch_size = m_config['batch_size']
    patience = m_config.get('patience', -1)
    pen_l2 = m_config['pen_l2']
    pen_constr = m_config['pen_constr']
    # Set the colocation set size for param definition
    # Colocation set
    pen_constr['coloc_set_size'] = mSampleLog.xTrainExtra[1].shape[0]

    nn_params_temp = m_config['nn_params']
    if 'side_info' in extra_args:
        nn_params_temp['type_sideinfo'] = extra_args['side_info']

    num_gradient_iterations = m_config['num_gradient_iterations']
    freq_accuracy = m_config['freq_accuracy']
    freq_save = m_config['freq_save']

    # Build the hyper params NN structure
    mParamsNN_list = [HyperParamsNN(model_name=model_name, n_state=int(mSampleLog.n_state), n_control=int(mSampleLog.n_control), actual_dt=float(mSampleLog.actual_dt),
                                    nn_params=nn_params_temp, optimizer=opt_info, seed_init=int(seed_val), qp_base=mSampleLog.qp_base, batch_size=int(batch_size),
                                    pen_l2=float(pen_l2), pen_constr=pen_constr, num_gradient_iterations=int(num_gradient_iterations),
                                    freq_accuracy=freq_accuracy, freq_save=int(freq_save), patience=patience)
                      for seed_val in seed_list]

    return weight_noise, bias_noise, constraint_noise, perform_gp_denoising, optimize_hyperparams, big_denoising, default_sigma_n, mSampleLog, mParamsNN_list, (type_baseline, data_set_file, out_file)

def denoise_trajectories(xTrainList, xnextTrainList, trajectory_length, n_rollout, optimize_hyperparams, big_denoising, default_sigma_n):
    trajectories = []
    for i in range(0, len(xTrainList), trajectory_length):
        trajectories.append(np.array(xTrainList[i:i+trajectory_length]))
    # not_in_xtrainlist = xnextTrainList[-1][-1][-trajectory_futures:]
    trajectory_max_idx = trajectory_length
    for idx in range(0, len(trajectories)):
        for future in range(0, n_rollout):
            trajectories[idx] = np.vstack((trajectories[idx], xnextTrainList[future][trajectory_max_idx-1]))
        trajectory_max_idx += trajectory_length
    
    sigma_f = 1
    sigma_l = 1
    sigma_n = default_sigma_n
    if optimize_hyperparams:
        sigma_f, sigma_l, sigma_n = gpr.find_optim_hyperparams(trajectories[0], big_denoising)

    print("DENOISING TRAJECTORIES...")
    print("sigma_f", sigma_f)
    print("sigma_l", sigma_l)
    print("sigma_n", sigma_n)
    print("big_denoising", big_denoising)
    trajectories = [gp_denoise_trajectory(trajectory, sigma_f, sigma_l, sigma_n, big_denoising) for trajectory in trajectories]

    xTrainList = [trajectory[:trajectory_length] for trajectory in trajectories]
    xTrainList = np.concatenate(xTrainList)

    xnextTrainList = [np.concatenate([trajectory[i:trajectory_length+i] for trajectory in trajectories]) for i in range(1, n_rollout+1)]
    xnextTrainList = np.array(xnextTrainList)

    return xTrainList, xnextTrainList


def gp_denoise_trajectory(trajectory, sigma_f, sigma_l, sigma_n, big_denoising):
    # noise must be specified if optimize_hyperparam is set to false
    return gpr.get_denoised_traj(trajectory, sigma_f, sigma_l, sigma_n, big_denoising)


def main_fn(path_config_file, extra_args={}):
    # Read the configration file
    weight_noise, bias_noise, constraint_noise, perform_gp_denoising, optimize_hyperparams, big_denoising, default_sigma_n, mSampleLog, mParamsNN_list, (type_baseline, data_set_file, out_file) = load_config_file(
        path_config_file, extra_args)
    reducedSampleLog = SampleLog(xTrain=None, xTrainExtra=None, uTrain=None, xnextTrain=None,
                                 lowU_train=mSampleLog.lowU_train, highU_train=mSampleLog.highU_train,
                                 xTest=None, xTestExtra=None, uTest=None, xnextTest=None,
                                 lowU_test=mSampleLog.lowU_test, highU_test=mSampleLog.highU_test,
                                 env_name=mSampleLog.env_name, env_extra_args=mSampleLog.env_extra_args,
                                 m_rng=mSampleLog.m_rng, seed_number=mSampleLog.seed_number, qp_indx=mSampleLog.qp_indx,
                                 qp_base=mSampleLog.qp_base, n_state=mSampleLog.n_state, n_control=mSampleLog.n_control,
                                 actual_dt=mSampleLog.actual_dt, disable_substep=mSampleLog.disable_substep,
                                 control_policy=mSampleLog.control_policy, n_rollout=mSampleLog.n_rollout)
    # Print the sample log info
    print("###### SampleLog ######")
    print(reducedSampleLog)

    # Training data
    xTrainList = np.asarray(mSampleLog.xTrain)
    (xTrainExtra, coloc_set) = mSampleLog.xTrainExtra
    xnextTrainList = np.asarray(mSampleLog.xnextTrain)
    (_, num_traj_data, trajectory_length) = mSampleLog.disable_substep
    coloc_set = jnp.array(coloc_set)
    if perform_gp_denoising:
        xTrainList, xnextTrainList = denoise_trajectories(xTrainList, xnextTrainList, trajectory_length, 5, optimize_hyperparams, big_denoising, default_sigma_n)
        print("Gaussian Process Denoising enabled")
    else:
        print("Gaussian Process Denoising disabled")

    # Testing data
    xTest = jnp.asarray(mSampleLog.xTest)
    xTestNext = jnp.asarray(mSampleLog.xnextTest)

    # DIctionary for logging training details
    final_log_dict = {i: {} for i in range(len(num_traj_data))}
    final_params_dict = {i: {} for i in range(len(num_traj_data))}

    for i, n_train in tqdm(enumerate(num_traj_data),  total=len(num_traj_data)):
        # The total number of point in this trajectory
        total_traj_size = n_train*trajectory_length

        # Dictionary to save the loss and the optimal params per radom seed
        dict_loss_per_seed = {mParamsNN.seed_init: {}
                              for mParamsNN in mParamsNN_list}
        dict_params_per_seed = {
            mParamsNN.seed_init: None for mParamsNN in mParamsNN_list}

        for _, mParamsNN in tqdm(enumerate(mParamsNN_list), total=len(mParamsNN_list), leave=False):

            # Some logging execution
            dbg_msg = "###### Hyper parameters: {} ######\n".format(
                type_baseline)
            dbg_msg += "{}\n".format(mParamsNN)
            tqdm.write(dbg_msg)

            # Build the neural network, the update, loss function and paramters of the neural network structure
            # TODO: Here I must edit something related with the Lagrange
            rng_gen, opt, (params, m_pen_eq_k, m_pen_ineq_k, m_lagr_eq_k, m_lagr_ineq_k), pred_xnext,\
                loss_fun, update, update_lagrange, loss_fun_constr, train_with_constraints = build_learner(
                    mParamsNN, type_baseline, constraint_noise=constraint_noise)

            # Initialize the optimizer with the initial parameters
            opt_state = opt.init(params)

            # Get the frequency at which to evaluate on the testing
            high_freq_record_rg = int(
                mParamsNN.freq_accuracy[0]*mParamsNN.num_gradient_iterations)
            high_freq_val = mParamsNN.freq_accuracy[1]
            low_freq_val = mParamsNN.freq_accuracy[2]
            update_freq = jnp.array([(j % high_freq_val) == 0 if j <= high_freq_record_rg else ((j % low_freq_val) == 0 if j < mParamsNN.num_gradient_iterations-1 else True)
                                     for j in range(mParamsNN.num_gradient_iterations)])

            # Dictionary to save the loss, the rollout error, and the colocation accuracy
            dict_loss = {'total_loss_train': list(), 'rollout_err_train': list(), 'total_loss_test': list(
            ), 'rollout_err_test': list(), 'coloc_err_train': list(), 'coloc_err_test': list()}

            # Store the best parameters attained by the learning process so far
            best_params = None 				# Weights returning best loss over the training set
            best_test_loss = None 			# Best test loss over the training set
            best_train_loss = None 			# Best training loss
            best_constr_val = None 			# Constraint attained by the best params
            best_test_noconstr = None 		# Best mean squared error without any constraints
            # Iteration since the last best weights values (iteration in terms of the loss evaluation period)
            iter_since_best_param = None
            number_lagrange_update = 0 		# Current number of lagrangian and penalty terms updates
            number_inner_iteration = 0 		# Current number of iteration the latest update
            # Period over which no improvement yields the best solution
            patience = mParamsNN.patience
            tqdm.write('\t\tEarly stopping criteria = {}'.format(patience > 0))

            # Iterate to learn the weight of the neural network
            for step in tqdm(range(mParamsNN.num_gradient_iterations), leave=False):

                # Random key generator for batch data
                rng_gen, subkey = jax.random.split(rng_gen)
                idx_train = jax.random.randint(subkey, shape=(
                    mParamsNN.batch_size,), minval=0, maxval=total_traj_size)

                # Sample and put into the GPU memory the batch-size data
                xTrain, xTrainNext = jnp.asarray(xTrainList[idx_train, :]), jnp.asarray(
                    xnextTrainList[:, idx_train, :])

                # Sample batches of the colocation data sets
                rng_gen, subkey = jax.random.split(rng_gen)
                idx_coloc_train = jax.random.randint(subkey, shape=(
                    mParamsNN.batch_size,), minval=0, maxval=coloc_set.shape[0])
                batch_coloc = coloc_set[idx_coloc_train]

                # Update the parameters of the NN and the state of the optimizer
                params, opt_state = update(params, opt_state, xTrainNext, xTrain, pen_eq_k=m_pen_eq_k, pen_ineq_sq_k=m_pen_ineq_k,
                                           lagr_eq_k=m_lagr_eq_k if m_lagr_eq_k.shape[
                                               0] <= 0 else m_lagr_eq_k[idx_coloc_train],
                                           lagr_ineq_k=m_lagr_ineq_k if m_lagr_ineq_k.shape[
                                               0] <= 0 else m_lagr_ineq_k[idx_coloc_train],
                                           extra_args_colocation=(batch_coloc, None, None))

                # If it is time to evaluate the models do it
                if update_freq[number_inner_iteration]:
                    if not train_with_constraints:  # In case there is no constraints -> Try to also log the constraints incurred by the current neural structure
                        loss_tr, (spec_data_tr, coloc_ctr) = loss_fun_constr(
                            params, xTrainNext, xTrain)
                        loss_te, (spec_data_te, coloc_cte) = loss_fun_constr(
                            params, xTestNext, xTest, extra_args_colocation=(coloc_set, None, None))
                    else:
                        loss_tr, (spec_data_tr, coloc_ctr) = loss_fun(
                            params, xTrainNext, xTrain)
                        loss_te, (spec_data_te, coloc_cte) = loss_fun(params, xTestNext, xTest, pen_eq_k=m_pen_eq_k, pen_ineq_sq_k=m_pen_ineq_k,
                                                                      lagr_eq_k=m_lagr_eq_k, lagr_ineq_k=m_lagr_ineq_k,
                                                                      extra_args_colocation=(coloc_set, None, None))

                    # Log the value obtained by evaluating the current model
                    dict_loss['total_loss_train'].append(float(loss_tr))
                    dict_loss['total_loss_test'].append(float(loss_te))
                    dict_loss['rollout_err_train'].append(spec_data_tr)
                    dict_loss['rollout_err_test'].append(spec_data_te)
                    dict_loss['coloc_err_train'].append(coloc_ctr)
                    dict_loss['coloc_err_test'].append(coloc_cte)

                    # Initialize the parameters for the best model so far
                    if number_inner_iteration == 0:
                        best_params = params
                        best_test_noconstr = jnp.mean(
                            spec_data_te[:, 1])  # Rollout mean
                        best_test_loss = loss_te
                        best_train_loss = loss_tr
                        best_constr_val = coloc_cte
                        iter_since_best_param = 0

                    # Check if the validation metrics has improved
                    if loss_te < best_test_loss:
                        best_params = params
                        best_test_loss = loss_te
                        best_test_noconstr = jnp.mean(spec_data_te[:, 1])
                        best_constr_val = coloc_cte
                        best_train_loss = loss_tr if loss_tr < best_train_loss else best_train_loss
                        iter_since_best_param = 0
                    else:
                        best_train_loss = loss_tr if loss_tr < best_train_loss else best_train_loss
                        iter_since_best_param += 1

                # Period at which we save the models
                if number_inner_iteration % mParamsNN.freq_save == 0:

                    # Update for the current seed what is the lost and the best params found so far
                    dict_loss_per_seed[mParamsNN.seed_init] = dict_loss
                    final_log_dict[i] = dict_loss_per_seed

                    # Update the best params found so far
                    dict_params_per_seed[mParamsNN.seed_init] = best_params
                    final_params_dict[i] = dict_params_per_seed

                    # Create the log file
                    mLog = LearningLog(env_name=mSampleLog.env_name, env_extra_args=mSampleLog.env_extra_args, baseline=type_baseline,
                                       sampleLog=reducedSampleLog, nn_hyperparams=mParamsNN_list, loss_evol=final_log_dict, learned_weights=final_params_dict,
                                       data_set_file=data_set_file)

                    # Save the current information in the file and close the file
                    mFile = open(out_file+'.pkl', 'wb')
                    pickle.dump(mLog, mFile)
                    mFile.close()

                    # Debug message to bring on the consolde
                    dbg_msg = '[Iter[{}][{}], N={}]\t train : {:.6f} | Loss test : {:.6f}\n'.format(
                        step, number_lagrange_update, n_train, loss_tr, loss_te)
                    dbg_msg += '\t\tBest Loss Function : Train = {:.6f} | Test = {:.6f}, {:.6f} | Constr = {}\n'.format(
                        best_train_loss, best_test_loss, best_test_noconstr, best_constr_val.round(6))
                    dbg_msg += '\t\tPer rollout loss train : {} | Loss test : {}\n'.format(
                        spec_data_tr[:, 1].round(6), spec_data_te[:, 1].round(6))
                    dbg_msg += '\t\tPer rollout EQ Constraint Train : {} | Test : {}\n'.format(
                        spec_data_tr[:, 2].round(6), spec_data_te[:, 2].round(6))
                    dbg_msg += '\t\tPer rollout INEQ Constraint Train : {} | Test : {}\n'.format(
                        spec_data_tr[:, 3].round(6), spec_data_te[:, 3].round(6))
                    dbg_msg += '\t\tColocotion loss : {}\n'.format(
                        coloc_cte.round(8))
                    tqdm.write(dbg_msg)

                # Update the number of iteration
                number_inner_iteration += 1

                # If the patience periode has been violated, try to break
                if patience > 0 and iter_since_best_param > patience:

                    # Debug message
                    tqdm.write('####################### EARLY STOPPING [{}][{}] #######################'.format(
                        step, number_lagrange_update))
                    tqdm.write('Best Loss Function : Train = {:.6f} | Test = {:.6f}, {:.6f}'.format(
                        best_train_loss, best_test_loss, best_test_noconstr))

                    # If there is no constraints then break
                    if not train_with_constraints:
                        break

                    # Check if the constraints threshold has been attained
                    if best_constr_val[1] < mParamsNN.pen_constr['tol_constraint_eq'] and best_constr_val[2] < mParamsNN.pen_constr['tol_constraint_ineq']:
                        tqdm.write('Constraints satisfied: [eq =  {}], [ineq = {}]\n'.format(
                            best_constr_val[1], best_constr_val[2]))
                        tqdm.write(
                            '##############################################################################\n')
                        break

                    # If the constraints threshold hasn't been violated, update the lagrangian and penalty coefficients
                    m_pen_eq_k, m_pen_ineq_k, m_lagr_eq_k, m_lagr_ineq_k = \
                        update_lagrange(params, pen_eq_k=m_pen_eq_k, pen_ineq_sq_k=m_pen_ineq_k,
                                        lagr_eq_k=m_lagr_eq_k, lagr_ineq_k=m_lagr_ineq_k, extra_args_colocation=(coloc_set, None, None))

                    # Update the new params to be the best params
                    params = best_params

                    # UPdate the number of inner iteration since the last lagrangian terms update
                    number_inner_iteration = 0

                    # Update the number of lagrange update so far
                    number_lagrange_update += 1

                    # Reinitialize the optimizer
                    opt_state = opt.init(params)

                    # Some printing for debig
                    tqdm.write('Update Penalty : [eq = {:.4f}, lag_eq = {:.4f}] | [ineq = {:.4f}, lag_ineq = {:.4f}]'.format(
                        m_pen_eq_k, jnp.sum(jnp.abs(m_lagr_eq_k)), m_pen_ineq_k, jnp.sum(m_lagr_ineq_k)))

            # Once the algorithm has converged, collect and save the data and params
            dict_loss_per_seed[mParamsNN.seed_init] = dict_loss
            final_log_dict[i] = dict_loss_per_seed
            # Here we update the weights with the noise
            key = jax.random.PRNGKey(0)
            # print(best_params) # TODO: Remove this
            best_params_noisy = {}
            for it_key, it_value in best_params.items():
                best_params_noisy[it_key] = {}
                for w_or_b, w_or_b_value in it_value.items():
                    if w_or_b == 'w':
                        best_params_noisy[it_key][w_or_b] = add_gaussian_noise(key, w_or_b_value, scale=weight_noise) if weight_noise > 0.00001 else w_or_b_value
                    elif w_or_b == 'b':
                        best_params_noisy[it_key][w_or_b] = add_gaussian_noise(key, w_or_b_value, scale=bias_noise) if weight_noise > 0.00001 else w_or_b_value
            dict_params_per_seed[mParamsNN.seed_init] = best_params_noisy
            final_params_dict[i] = dict_params_per_seed

            # Cretae the log
            mLog = LearningLog(env_name=mSampleLog.env_name, env_extra_args=mSampleLog.env_extra_args, baseline=type_baseline,
                               sampleLog=reducedSampleLog, nn_hyperparams=mParamsNN_list, loss_evol=final_log_dict, learned_weights=final_params_dict,
                               data_set_file=data_set_file)

            # Save the current information
            mFile = open(out_file+'.pkl', 'wb')
            pickle.dump(mLog, mFile)
            mFile.close()

    # for i, xTrain, xTrainNext in tqdm(zip(range(len(xTrainList)), xTrainList, xnextTrainList),  total=len(xTrainList)):
    # 	dict_loss_per_seed = {mParamsNN.seed_init : {} for mParamsNN in mParamsNN_list}
    # 	dict_params_per_seed = {mParamsNN.seed_init : None for mParamsNN in mParamsNN_list}
    # 	for _, mParamsNN in tqdm(enumerate(mParamsNN_list), total=len(mParamsNN_list), leave=False):
    # 		dbg_msg = "###### Hyper parameters: {} ######\n".format(type_baseline)
    # 		dbg_msg += "{}\n".format(mParamsNN)
    # 		tqdm.write(dbg_msg)
    # 		# Build the neural network
    # 		rng_gen, opt, params, pred_xnext, loss_fun, update = build_learner(mParamsNN, type_baseline)
    # 		# Initialize teh optimizer
    # 		opt_state = opt.init(params)
    # 		dict_loss = {'total_loss_train' : list(), 'rollout_err_train' : list(), 'total_loss_test' : list(), 'rollout_err_test' : list()}
    # 		# Iterate to learn the weight of the neural network
    # 		for step in tqdm(range(mParamsNN.num_gradient_iterations), leave=False):
    # 			rng_gen, subkey = jax.random.split(rng_gen)
    # 			idx_train = jax.random.randint(subkey, shape=(mParamsNN.batch_size,), minval=0, maxval=xTrain.shape[0])
    # 			params, opt_state = update(params, opt_state, xTrainNext[:,idx_train,:], xTrain[idx_train,:])
    # 			if step % mParamsNN.freq_accuracy == 0:
    # 				loss_tr, spec_data_tr = loss_fun(params, xTrainNext[:,idx_train,:], xTrain[idx_train, :])
    # 				loss_te, spec_data_te = loss_fun(params, xTestNext, xTest)
    # 				dict_loss['total_loss_train'].append(float(loss_tr))
    # 				dict_loss['total_loss_test'].append(float(loss_te))
    # 				dict_loss['rollout_err_train'].append(spec_data_tr)
    # 				dict_loss['rollout_err_test'].append(spec_data_te)

    # 			if step % mParamsNN.freq_save == 0: # Latest loss function
    # 				dict_loss_per_seed[mParamsNN.seed_init] = dict_loss
    # 				final_log_dict[i] = dict_loss_per_seed
    # 				dict_params_per_seed[mParamsNN.seed_init] = params
    # 				final_params_dict[i] = dict_params_per_seed
    # 				mLog = LearningLog(env_name=mSampleLog.env_name, env_extra_args=mSampleLog.env_extra_args, baseline=type_baseline,
    # 					sampleLog=reducedSampleLog, nn_hyperparams= mParamsNN_list, loss_evol=final_log_dict, learned_weights=final_params_dict,
    # 					data_set_file=data_set_file)
    # 				# Save the current information
    # 				mFile = open(out_file+'.pkl', 'wb')
    # 				pickle.dump(mLog, mFile)
    # 				mFile.close()
    # 				dbg_msg = '[Iter {}, Loss]\t train : {:.6f} | Loss test : {:.6f}\n'.format(step, loss_tr, loss_te)
    # 				dbg_msg += '\t\tPer rollout loss train : {} | Loss test : {}\n'.format(spec_data_tr[:,1].round(6), spec_data_te[:,1].round(6))
    # 				dbg_msg += '\t\tPer rollout Constraint Train : {} | Test : {}\n'.format(spec_data_tr[:,2].round(6), spec_data_te[:,2].round(6))
    # 				tqdm.write(dbg_msg)
    # 		dict_loss_per_seed[mParamsNN.seed_init] = dict_loss
    # 		final_log_dict[i] = dict_loss_per_seed
    # 		dict_params_per_seed[mParamsNN.seed_init] = params
    # 		final_params_dict[i] = dict_params_per_seed
    # 		mLog = LearningLog(env_name=mSampleLog.env_name, env_extra_args=mSampleLog.env_extra_args, baseline=type_baseline,
    # 					sampleLog=reducedSampleLog, nn_hyperparams= mParamsNN_list, loss_evol=final_log_dict, learned_weights=final_params_dict,
    # 					data_set_file=data_set_file)
    # 		# Save the current information
    # 		mFile = open(out_file+'.pkl', 'wb')
    # 		pickle.dump(mLog, mFile)
    # 		mFile.close()

    # print(len(xTrain))
    # # Build the neural network
    # rng_gen, opt, params, pred_xnext, loss_fun, update = build_learner(mParamsNN, type_baseline)


if __name__ == "__main__":
    import time
    import argparse

    # Command line argument for setting parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='yaml configuration file for training/testing information: see reacher_cfg1.yaml for more information')
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--baseline', type=str, default='')
    parser.add_argument('--num_grad', type=int, default=0)
    parser.add_argument('--side_info', type=int, default=-1)
    args = parser.parse_args()
    m_config_aux = {'cfg': args.cfg}
    if args.output_file != '':
        m_config_aux['output_file'] = args.output_file
    if args.batch_size > 0:
        m_config_aux['batch_size'] = args.batch_size
    if args.baseline != '':
        m_config_aux['baseline'] = args.baseline
    if args.num_grad > 0:
        m_config_aux['num_gradient_iterations'] = args.num_grad
    if args.input_file != '':
        m_config_aux['train_data_file'] = args.input_file
    if args.side_info >= 0:
        m_config_aux['side_info'] = args.side_info

    main_fn(args.cfg, m_config_aux)
