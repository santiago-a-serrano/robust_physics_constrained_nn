import math
import pickle
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
from train import build_learner
import numpy as np

def enforce_max_norm(vector, max_norm):
    vector_norm = jnp.linalg.norm(vector)
    if vector_norm <= max_norm:
        return vector
    else:
        normalized_vector = vector / vector_norm
        return normalized_vector * max_norm
    
def enforce_max_norm_matrix(matrix, max_norm):
    matrix_norm = jnp.linalg.norm(matrix, 'fro')
    if matrix_norm <= max_norm:
        return matrix
    else:
        normalized_matrix = matrix / matrix_norm
        return normalized_matrix * max_norm


def loss_fn(x_noise, loss_fun_1, loss_fun_2, params, x_next, x, max_norm):
    batch_size = len(x)
    x_noise = enforce_max_norm(x_noise, max_norm)
    noise_for_batch = jnp.stack([x_noise for _ in range(batch_size)])
    term_1, _ = loss_fun_1(params, x_next, x + noise_for_batch)
    term_2, _ = loss_fun_2(params, x_next, x + noise_for_batch)
    return term_1 + term_2

def loss_fn_traj(traj_noise, loss_fun_1, loss_fun_2, params, x_next, x, max_norm):
    traj_noise = enforce_max_norm_matrix(traj_noise, max_norm)
    term_1, _ = loss_fun_1(params, x_next, x + traj_noise)
    term_2, _ = loss_fun_2(params, x_next, x + traj_noise)
    return term_1 + term_2

def update(x_noise, loss_fun_1, loss_fun_2, params, x_next, x, opt_state, optimizer, max_norm, make_traj_noise=False):
    def loss_fn_wrapped(x_noise):
        if make_traj_noise:
            return loss_fn_traj(x_noise, loss_fun_1, loss_fun_2, params, x_next, x, max_norm)
        else:
            return loss_fn(x_noise, loss_fun_1, loss_fun_2, params, x_next, x, max_norm)

    grad_fn = jax.value_and_grad(loss_fn_wrapped)
    loss, gradients = grad_fn(x_noise)
    # negate the gradients to maximize the loss
    updates, new_opt_state = optimizer.update(-gradients, opt_state)
    new_x_noise = optax.apply_updates(x_noise, updates)
    return new_x_noise, new_opt_state, loss

def random_vector_of_norm(norm, shape):
    vector = jax.random.normal(jax.random.PRNGKey(0), shape)
    vector_norm = jnp.linalg.norm(vector)
    normalized_vector = vector / vector_norm
    return normalized_vector * norm

def random_matrix_of_frobenius_norm(norm, shape):
    matrix = jax.random.normal(jax.random.PRNGKey(0), shape)
    matrix_norm = jnp.linalg.norm(matrix, 'fro')
    normalized_matrix = matrix / matrix_norm
    return normalized_matrix * norm
    

# make_traj_noise = False -> noise of shape (datapoint_size,)
# make_traj_noise = True -> noise of shape (trajectory_length, datapoint_size)
# TODO: Implement noising of weights and biases
def main(max_x_noise, trained_model_path, trajectories_path, max_weight_noise=0, max_bias_noise=0, make_traj_noise=False):
    batch_size = 64
    datapoint_size = 4
    trajectory_length = 300
    max_traj_noise = max_x_noise * math.sqrt(trajectory_length)

    with open(trained_model_path, 'rb') as f:
        trained_model = pickle.load(f)
        # TODO: Make the 4 and 101 dynamic parameters
        idx = len(trained_model.learned_weights) - 1
        seed = 101
        params = trained_model.learned_weights[idx][seed]
    _, _, _, pred_xnext, loss_fun, _, _, loss_fun_constr, _ = build_learner(trained_model.nn_hyperparams[0], trained_model.baseline)

    if make_traj_noise:
        x_noise = random_matrix_of_frobenius_norm(max_traj_noise, (trajectory_length, datapoint_size))
    else:
        x_noise = random_vector_of_norm(max_x_noise, (datapoint_size,))

    optimizer = optax.adam(learning_rate=0.001)

    with open(trajectories_path, 'rb') as f:
        generated_data = pickle.load(f)
        x = generated_data.xTrain
        # This are the next 5 states of any state in xtrain
        xnext = generated_data.xnextTrain

    # Initialize the optimizer state
    opt_state = optimizer.init(x_noise)
    real_x_noise = None

    if make_traj_noise:
        print("make_traj_noise was set to true, making adv. noise for complete trajectory")
        print(x_noise)
        x_split = jnp.array(jnp.array_split(x, len(x) / trajectory_length))
        xnext_split = jnp.array([jnp.array_split(xnext_array, len(xnext_array) / trajectory_length) for xnext_array in xnext])
        for trajectory_id in range(len(x_split)):
            x_noise, opt_state, loss = update(x_noise, loss_fun, loss_fun_constr, params, xnext_split[:, trajectory_id], x_split[trajectory_id], opt_state, optimizer, max_traj_noise, make_traj_noise=True)
            print("Trajectory:", trajectory_id, "\tLoss:", loss)
            # the traj_noise before enforcing the max norm
            print("sub_traj_noise[0]:", x_noise[0], "norm:", jnp.linalg.norm(x_noise, 'fro') / math.sqrt(trajectory_length))
            real_traj_noise = enforce_max_norm_matrix(x_noise, max_traj_noise)
            print("real_traj_noise[0]", real_traj_noise[0], "norm:", jnp.linalg.norm(real_traj_noise, 'fro') / math.sqrt(trajectory_length))
            print()

        np.save(f'generated/worst_perturbations/{max_x_noise}trajmaxnorm.npy', real_traj_noise)
        print("Saved worst perturbation for", max_x_noise, "max traj noise norm in data.")

    else:
        print("make_traj_noise was set to false, making adv. noise for only one datapoint")
        for epoch in range(100):
            # Select a random batch of size batch_size from x and xnext
            indices = jax.random.choice(jax.random.PRNGKey(epoch), len(x), shape=(batch_size,), replace=False)
            x_batch = x[indices]
            xnext_batch = xnext[:, indices]
            x_noise, opt_state, loss = update(x_noise, loss_fun, loss_fun_constr, params, xnext_batch, x_batch, opt_state, optimizer, max_x_noise, make_traj_noise=False)
            print("Epoch:", epoch, "\tLoss:", loss)
            # the x_noise before enforcing the max norm
            print("sub_x_noise:", x_noise, "norm:", jnp.linalg.norm(x_noise))
            # the x_noise after enforcing the max norm
            real_x_noise = enforce_max_norm(x_noise, max_x_noise)
            print("real_x_noise", real_x_noise, "norm:", jnp.linalg.norm(real_x_noise))
            print()

        np.save(f'generated/worst_perturbations/{max_x_noise}datamaxnorm.npy', real_x_noise)
        print("Saved worst perturbation for", max_x_noise, "max noise norm in data.")


# SPECIFY PARAMETERS HERE:
main(max_x_noise=0.2, 
     trained_model_path='pregenerated/trained_models/mlp/no_noise.pkl', 
     trajectories_path='pregenerated/trajectories/gaussian_noise/no_noise.pkl', 
     make_traj_noise=True)