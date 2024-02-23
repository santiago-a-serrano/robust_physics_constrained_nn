import pickle

with open('FILEPATHHERE.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
