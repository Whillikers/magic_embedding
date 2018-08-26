import os
import pickle
from glob import glob
import click
import numpy as np
from scipy.sparse import lil_matrix, save_npz


@click.command()
@click.option('--coding', default='5000', help='number of cards in encoding')
@click.option('--out', default='datasets/matrix/card_deck_csr.npz')
@click.option('--dup', default=True, help='count duplicate cards in a deck')
@click.option('--sideboard', default=False, help='count cards in sideboards')
@click.option('--verbose', default=True)
@click.argument('dir_decks', default='datasets/raw_decks/deckbox')
def save_csr(coding, out, dup, sideboard, verbose, dir_decks):
    '''
    Save a list of decks as a CSR matrix (efficient row access) where rows are
    cards and columns are decks.
    '''
    with open('datasets/code_mappings/decoding_' + coding + '.pkl', 'rb') as f:
        decoding = pickle.load(f)
    with open('datasets/code_mappings/encoding_' + coding + '.pkl', 'rb') as f:
        encoding = pickle.load(f)
    paths = glob(os.path.join(dir_decks, '*_mainboard.txt'))
    out = out.replace('.npz', '_' + coding + '.npz')

    n_cards = len(decoding)
    n_decks = len(paths)

    matrix = lil_matrix((n_cards, n_decks), dtype=np.int8)
    print('SHAPE:', matrix.shape)

    if verbose:
        print('Making LIL matrix...')

    for i_deck, path in enumerate(paths):
        if verbose and i_deck % 10000 == 0:
            print('On deck: {}'.format(i_deck))

        sideboard_path = path.replace('_mainboard.txt', '_sideboard.txt')
        with open(path, 'r') as f:
            lines = f.readlines()

        if sideboard and os.path.exists(sideboard_path):
            with open(sideboard_path, 'r') as f:
                lines.extend(f.readlines())

        for line in lines:
            count, name = line.strip().split(' ', 1)

            try:
                code = encoding[name]
            except KeyError:
                code = 0  # Unknown name; map to UNK

            if dup:
                count = int(count)
                if matrix[code, i_deck]:  # Sideboard duplicate of mainboard
                    matrix[code, i_deck] += count
                else:
                    matrix[code, i_deck] = count
            else:
                matrix[code, i_deck] = 1

    if verbose:
        print('Done making LIL matrix; converting to CSR and saving...')

    save_npz(out, matrix.tocsr(), compressed=True)

    if verbose:
        print('Done!')


if __name__ == '__main__':
    save_csr()
