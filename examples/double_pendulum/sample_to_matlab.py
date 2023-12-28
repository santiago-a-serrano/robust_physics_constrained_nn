import pickle
import numpy as np
import scipy.io

fileName = '0.045noise'

with open(f'sindypi/data/{fileName}.pkl', 'rb') as f:
    sampleLog = pickle.load(f)

# TODO: The first and last derivatives are still different from the sample. Maybe there's another method.
Data = {'Data': sampleLog.xTrain, 
        'Data_test': sampleLog.xTest[:3001], 
        'dData': np.gradient(sampleLog.xTrain, axis=0) / 0.001, 
        'dData_test': np.gradient(sampleLog.xTest[:3001], axis=0) / 0.001}
with open(f'sindypi/mats/{fileName}.mat', 'wb') as f:
    scipy.io.savemat(f, Data)
    # TODO: timestep is 0.001
