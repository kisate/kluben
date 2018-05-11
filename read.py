import pickle
with open('img.txt', 'rb') as f:
    data_new = pickle.load(f)
    print(data_new)