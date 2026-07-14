import pickle

def save_history(history, filepath):
    # save_history(history, "history/initial_resnet.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(history.history, f)

def load_history(filepath):
    # load_history("history/initial_resnet.pkl")
    with open(filepath, "rb") as f:
        return pickle.load(f)