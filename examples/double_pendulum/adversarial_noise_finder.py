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
# TODO: Fix the noise norm issue. Print the loss.
def loss_fn(x_noise, loss_fun_1, loss_fun_2, params, x_next, x, max_norm):
    batch_size = len(x)
    x_noise = enforce_max_norm(x_noise, max_norm)
    noise_for_batch = jnp.stack([x_noise for _ in range(batch_size)])
    term_1, _ = loss_fun_1(params, x_next, x + noise_for_batch)
    term_2, _ = loss_fun_2(params, x_next, x + noise_for_batch)
    return term_1 + term_2

def update(x_noise, loss_fun_1, loss_fun_2, params, x_next, x, opt_state, optimizer, max_norm):
    def loss_fn_wrapped(x_noise):
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



def main(max_x_noise, max_weight_noise=0, max_bias_noise=0):
    batch_size = 64
    datapoint_size = 4

    with open('DEST_FILE/base_datatrain_00prcnt.pkl', 'rb') as f:
        trained_model = pickle.load(f)
        # TODO: Make the 4 and 101 dynamic parameters
        idx = len(trained_model.learned_weights) - 1
        seed = 101
        params = trained_model.learned_weights[idx][seed]
    _, _, _, pred_xnext, loss_fun, _, _, loss_fun_constr, _ = build_learner(trained_model.nn_hyperparams[0], trained_model.baseline)

    # TODO: Implement weight and bias noise too
    x_noise = random_vector_of_norm(max_x_noise, (datapoint_size,))
    optimizer = optax.adam(learning_rate=0.001)

    with open('DEST_FILE/datatrain_noise00prcnt.pkl', 'rb') as f:
        generated_data = pickle.load(f)
        x = generated_data.xTrain
        # This are the next 5 states of any state in xtrain
        xnext = generated_data.xnextTrain

    # TODO: Remove this (it's for testing purposes)
    indices_test = jax.random.choice(jax.random.PRNGKey(99), len(x), shape=(batch_size,), replace=False)
    r_x_noise_to_add = jnp.asarray([0.01401838, -0.0105391,  -0.00600591,  0.00750609])
    result = loss_fn(r_x_noise_to_add, loss_fun, loss_fun_constr, params, xnext[:, indices_test], x[indices_test], 999999)
    print("loss", result)

    # Initialize the optimizer state
    opt_state = optimizer.init(x_noise)
    real_x_noise = None

    # TODO: More epochs?
    # TODO: Only 0.2 max_noise was trained 200 epochs.
    for epoch in range(200):
        # Select a random batch of size batch_size from x and xnext
        indices = jax.random.choice(jax.random.PRNGKey(epoch), len(x), shape=(batch_size,), replace=False)
        x_batch = x[indices]
        xnext_batch = xnext[:, indices]
        x_noise, opt_state, loss = update(x_noise, loss_fun, loss_fun_constr, params, xnext_batch, x_batch, opt_state, optimizer, max_x_noise)
        print("Epoch:", epoch, "\tLoss:", loss)
        # the x_noise before enforcing the max norm
        print("sub_x_noise:", x_noise, "norm:", jnp.linalg.norm(x_noise))
        # the x_noise after enforcing the max norm
        real_x_noise = enforce_max_norm(x_noise, max_x_noise)
        print("real_x_noise", real_x_noise, "norm:", jnp.linalg.norm(real_x_noise))
        print()

    np.save(f'DEST_FILE/worst_perturbations/{max_x_noise}datamaxnorm.npy', real_x_noise)
    print("Saved worst perturbation for", max_x_noise, "max noise norm in data.")

    



# main(max_x_noise=0.01)
# main(max_x_noise=0.03)
# main(max_x_noise=0.05)
# main(max_x_noise=0.1)
# main(max_x_noise=0.2)