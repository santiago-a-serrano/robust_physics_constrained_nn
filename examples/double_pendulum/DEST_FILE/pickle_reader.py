import pickle

with open('datatrain.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
