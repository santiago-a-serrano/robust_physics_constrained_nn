import pickle
import pysindy as ps
import numpy as np

import jax
import jax.numpy as jnp

from pysindy.feature_library import PolynomialLibrary, FourierLibrary, ConcatLibrary
from train import denoise_trajectories

class SINDy:
    def __init__(self, data_file_path, gpsindy=False):
        data_file = open(data_file_path, 'rb')
        mSampleLog = pickle.load(data_file)
        data_file.close()
            # Training data
        xTrainList = np.asarray(mSampleLog.xTrain)
        (xTrainExtra, coloc_set) = mSampleLog.xTrainExtra
        xnextTrainList = np.asarray(mSampleLog.xnextTrain)
        (_, num_traj_data, trajectory_length) = mSampleLog.disable_substep
        coloc_set = jnp.array(coloc_set)
        if gpsindy:
            xTrainList, xnextTrainList = denoise_trajectories(xTrainList, xnextTrainList, trajectory_length, 5, True, False, -1)
            print("Gaussian Process Denoising enabled")
        else:
            print("Gaussian Process Denoising disabled")

        # Define your libraries
        poly_library = PolynomialLibrary()
        fourier_library = FourierLibrary(n_frequencies=1)  # Adjust n_frequencies as needed

        # Combine the libraries
        combined_library = ConcatLibrary([poly_library, fourier_library])
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=0), feature_library=combined_library)
        xTrainListSep = [xTrainList[i:i+300] for i in range(0, len(xTrainList), 300)]

        model.fit(xTrainListSep, multiple_trajectories=True)
        
        self.model = model


    def predict(self, trajectory, traj_len):
        t = np.arange(traj_len)
        return self.model.simulate(trajectory, t)