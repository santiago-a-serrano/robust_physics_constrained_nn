import pickle
import jax.numpy as jnp
import numpy as np
from train import build_learner
import torch.nn as nn
from torch import optim
import torch

class AdversarialPerturbationOptimizer(nn.Module):
    def random_vector_of_norm(self, norm, shape):
        vector = torch.randn(shape)
        vector_norm = torch.linalg.vector_norm(vector)
        normalized_vector = vector / vector_norm
        return normalized_vector * norm

    def enforce_max_norm(self, vector, max_norm):
        vector_norm = torch.linalg.vector_norm(vector)
        if vector_norm <= max_norm:
            return vector
        else:
            normalized_vector = vector / vector_norm
            return normalized_vector * max_norm
            

    # original_params is params without any noise or modification
    def __init__(self, max_x_noise, batch_size=64, n_x=4):
        super().__init__()
        with open('DEST_FILE/base_datatrain_00prcnt.pkl', 'rb') as f:
            trained_model = pickle.load(f)
            # TODO: Make the 4 and 101 dynamic parameters
            idx = len(trained_model.learned_weights) - 1
            seed = 101
            params = trained_model.learned_weights[idx][seed]
        _, _, _, pred_xnext, loss_fun, _, _, loss_fun_constr, _ = build_learner(trained_model.nn_hyperparams[0], trained_model.baseline)

        self.params = params
        self.loss_fun_1 = loss_fun
        self.loss_fun_2 = loss_fun_constr
        self.batch_size = batch_size
        self.max_x_noise = max_x_noise

        self.x_noise = nn.Parameter(self.random_vector_of_norm(max_x_noise, (n_x,)))
        # TODO: Implement weight and bias noise

    # x should be one batch of data (size 64)
    # xnext should be one batch too (size 64). Should be real data, not inferred by the model.
    def forward(self, x_next, x):
        # TODO: Add the rest of parameters, such as extra_args, pen_eq_k, etc. so that we also have support for physics-constrained/informed networks.
        self.x_noise = self.enforce_max_norm(self.x_noise, self.max_x_noise)
        noise_for_batch = torch.stack([self.x_noise for _ in range(self.batch_size)])
        compound_loss = self.loss_fun_1(self.params, x_next, x + noise_for_batch) + self.loss_fun_2(self.params, x_next, x + noise_for_batch)
        return -compound_loss # since we want to maximize it

def main(max_x_noise, max_weight_noise=0, max_bias_noise=0):
    batch_size = 64
    model = AdversarialPerturbationOptimizer(max_x_noise, batch_size, 4)
    optimizer = optim.Adam(model.parameters())

    with open('DEST_FILE/datatrain_noise00prcnt.pkl', 'rb') as f:
        generated_data = pickle.load(f)
        x = torch.from_numpy(np.asarray(generated_data.xTrain).copy())
        x_batches = torch.split(x, batch_size)
        # This are the next 5 states of any state in xtrain
        xnext = torch.from_numpy(np.asarray(generated_data.xnextTrain).copy())
        xnext_batches = torch.split(xnext, batch_size)

    # wp is short for worst perturbation
    wp = None
    wp_loss = None
    wp_norm = None

    # TODO: More epochs?
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(xnext_batches[epoch], x_batches[epoch])
        if wp_loss == None or output > wp_loss:
            wp = model.x_noise.item()
            wp_loss = -output.item()
            wp_norm = torch.linalg.vector_norm(wp)
        print("Epoch", epoch)
        print("Perturbation Loss:", -output.item(), "\t", "WP Loss:", wp_loss)
        print("Perturbation:", model.x_noise.item(), "\t", "WP:", wp)
        print("Perturbation Norm:", torch.linalg.vector_norm(model.x_noise.item()), "\t", "WP Norm", wp_norm)
        output.backward()
        optimizer.step()
        print()




    



    


    




main(max_x_noise=0.02)
