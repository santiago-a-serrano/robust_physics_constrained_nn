import sys

sys.path.append('/usr/local/cuda-11.8/lib64')

import torch.nn as nn
import torch
from torch import optim

def squared_exponential_kernel(argument_1, argument_2, sigma_f, sigma_l):
    return (sigma_f * torch.exp(-(torch.norm(argument_1 - argument_2)**2) / (2 * sigma_l**2)))

# X1 and X2 are sets of points
def cov_matrix(X1, X2, covariance_function, sigma_f, sigma_l):
    return torch.stack([torch.stack([covariance_function(x1, x2, sigma_f, sigma_l) for x1 in X1]) for x2 in X2])
    
def find_optim_hyperparams(trajectory, big_denoising=False):
    t = torch.arange(0, len(trajectory), dtype=torch.float)
    trajectory = torch.from_numpy(trajectory).T
    model = LogMarginalLikelihood(trajectory, t, big_denoising)
    optimizer = optim.Rprop(model.parameters())
    print("OPTIMIZING HYPERPARAMETERS with optimizer", optimizer)
    epoch = 0

    best_lml = None
    best_f = None
    best_l = None
    best_n = None

    # TODO: Implement better stopping criteria?
    while epoch < 100:
        optimizer.zero_grad()
        output = model() # TODO: Do I have to pass parameters of some sort?
        if best_lml == None or output < best_lml:
            best_lml = output.item()
            best_f = model.theta_f.item()
            best_l = model.theta_l.item()
            best_n = model.theta_n.item()
        print("Epoch", epoch)
        print("LML:", output.item(), "\t", "best_lml:", best_lml)
        print("theta_f", model.theta_f.item(), "\t", "best_f:", best_f)
        print("theta_l", model.theta_l.item(), "\t", "best_l:", best_l)
        print("theta_n", model.theta_n.item(), "\t", "best_n:", best_n)
        output.backward()
        optimizer.step()
        epoch += 1
        print()

    return best_f, best_l, best_n

    
class LogMarginalLikelihood(nn.Module):
    def __init__(self, trajectory, t, big_denoising=False):
        super().__init__()
        self.theta_f = nn.Parameter(torch.tensor(1.0))
        self.theta_l = nn.Parameter(torch.tensor(1.0))
        self.theta_n = nn.Parameter(torch.tensor(0.1))
        self.trajectory = trajectory
        self.t = t
        self.big_denoising = big_denoising
    def forward(self, n=4): # the state of the system has 4 dimensions for the case of the double pendulum example
        lml = None
        K = cov_matrix(self.t, self.t, squared_exponential_kernel, self.theta_f, self.theta_l)
        if self.big_denoising:
            first_multiplication = self.trajectory @ torch.inverse(K + (3e-7 + self.theta_n) * torch.eye(len(self.t)))
            second_multiplication = first_multiplication @ self.trajectory.T
            lml = 0.5 * torch.trace(second_multiplication) + \
                0.5 * torch.logdet(K + (3e-7 + self.theta_n) * torch.eye(len(self.t))) + 0.5 * n * torch.log(torch.tensor(2 * torch.pi))
        else:
            first_multiplication = self.trajectory @ torch.inverse(K + (3e-7 + self.theta_n**2) * torch.eye(len(self.t)))
            second_multiplication = first_multiplication @ self.trajectory.T
            lml = 0.5 * torch.trace(second_multiplication) + \
                0.5 * torch.logdet(K + (3e-7 + self.theta_n**2) * torch.eye(len(self.t))) + 0.5 * n * torch.log(torch.tensor(2 * torch.pi))
        
        return lml


def get_denoised_traj(trajectory, sigma_f, sigma_l, sigma_n, big_denoising=False):
    t = torch.arange(0, len(trajectory), dtype=torch.float)
    trajectory = torch.from_numpy(trajectory).T
    K = cov_matrix(t, t, squared_exponential_kernel, sigma_f, sigma_l)
    if big_denoising:
        covariance_matrix_with_noise = (K + (3e-7 + sigma_n) * torch.eye(len(t)))
    else:
        covariance_matrix_with_noise = (K + (3e-7 + sigma_n**2) * torch.eye(len(t)))
    # Mean.
    mean_at_values = torch.mm(
        K,
        torch.mm(trajectory,
                torch.inverse(covariance_matrix_with_noise).T).T)

    return mean_at_values.numpy()
