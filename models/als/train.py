import pickle
import numpy as np
import implicit
from scipy.sparse import load_npz


def train_als(factors=16, reg=0.01, n_cards=5000):
    path_csr = 'datasets/matrix/card_deck_csr_' + str(n_cards) + '.npz'

    with open('datasets/code_mappings/decoding_' +
              str(n_cards) + '.pkl', 'rb') as f:
        decoding = pickle.load(f)
    with open('datasets/code_mappings/encoding_' +
              str(n_cards) + '.pkl', 'rb') as f:
        encoding = pickle.load(f)

    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 regularization=reg,
                                                 use_gpu=False,
                                                 calculate_training_loss=True)

    matrix = load_npz(path_csr)
    model.fit(matrix)
    #  np.save('models/als/saved/deck_factors.npy', model.user_factors)
    #  np.save('models/als/saved/card_factors.npy', model.item_factors)
