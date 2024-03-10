# Utility function to print a pickle file through console

import pickle

with open('FILEPATHHERE.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
