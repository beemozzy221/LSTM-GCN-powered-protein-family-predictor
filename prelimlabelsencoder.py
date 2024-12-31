import numpy as np

class LabelsEncode:

    def __init__(self, pids:np.ndarray):
        self.pids=pids
    
    def labels2onehot(self):
        vocab_prelim_size = len(self.pids)
        vocab_prelim = dict(zip(self.pids, range(vocab_prelim_size)))
        vocab_size = len(vocab_prelim)
        vocab_embed = dict(zip(vocab_prelim.keys(), range(vocab_size)))
        
        # Convert vocab to one-hot
        vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
        for _, val in vocab_embed.items():
            vocab_one_hot[val, val] = 1
        embed_y = [vocab_embed[v] for v in self.pids]
        fam_y = np.array([vocab_one_hot[j, :] for j in embed_y])

        return fam_y




