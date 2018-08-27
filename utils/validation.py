'''
Utilities for evaluating performance on a test set.
'''
from random import sample
import numpy as np
from scipy.sparse import lil_matrix


def train_val_split(matrix, val_frac=0.2):
    '''
    Split a sparse user-item matrix into:
     - The input matrix, with some decks missing cards
     - A matrix mapping the decks (columns) to those missing cards (rows)
     - A list of deck ids effected
    '''
    train = matrix.tolil(copy=True)
    test = lil_matrix(train.shape, dtype=bool)
    test[test != 0] = 1

    np.random.seed(1337)
    nz_x, nz_y = train.nonzero()
    nz_ids = list(zip(nz_x, nz_y))
    num_val = int(len(nz_ids) * val_frac)

    val_points = sample(nz_ids, num_val)
    masked_cards, masked_decks = zip(*val_points)
    train[masked_cards, masked_decks] = 0
    test[masked_cards, masked_decks] = 1

    train = train.tocsr()
    return train.tocsr(), test.tocsc(), list(set(masked_decks))


def top_n_frac(model, train_matrix, val_matrix, deck_ids, n=30):
    '''
    Counts the average fraction of validation cards removed from a deck
    that are included in the model's top n recommendations for that deck.
    '''
    train_matrix_lookup = train_matrix.transpose().tocsr()
    frac = 0
    for i, deck_id in enumerate(deck_ids):
        val_cards = set(np.where(val_matrix[:, deck_id].toarray())[0])
        rec = model.recommend(deck_id, train_matrix_lookup, N=n)
        rec_cards = set(list(zip(*rec))[0])
        common = rec_cards.intersection(val_cards)
        frac += len(common) / len(val_cards)

    frac /= len(deck_ids)
    return frac
