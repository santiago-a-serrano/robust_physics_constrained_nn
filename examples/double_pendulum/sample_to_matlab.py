import pickle
import numpy as np
import scipy.io

fileName = 'noise10prcnt'

with open(f'sindypi/data/datatrain_{fileName}.pkl', 'rb') as f:
    sampleLog = pickle.load(f)

# TODO: The first and last derivatives are still different from the sample. Maybe there's another method.
Data = {'Data': sampleLog.xTrain[:300], 
        'Data_test': sampleLog.xTest[:300], 
        'dData': np.gradient(sampleLog.xTrain[:300], axis=0) / 0.01, 
        'dData_test': np.gradient(sampleLog.xTest[:300], axis=0) / 0.01}
with open(f'sindypi/mats/{fileName}.mat', 'wb') as f:
    scipy.io.savemat(f, Data)
    # TODO: timestep is 0.001
