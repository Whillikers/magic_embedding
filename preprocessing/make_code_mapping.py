import os
from collections import OrderedDict
import pickle
import click


@click.command()
@click.option('--n', default=None, help='number of cards to map', type=int)
@click.option('--out_dir', default='datasets/code_mappings/')
@click.argument('path_counts', default='datasets/counts/cards_and_counts.pkl')
def make_code_mapping(n, out_dir, path_counts):
    '''
    Make a code mapping for at most n cards, using a preexisting count index
    pickle to measure popularity (see: count_cards.py).
    UNK corresponds to all cards not in the top n by count.
    '''
    if not n:
        n = 'all'
    suffix = '_' + str(n) + '.pkl'

    card_to_code = OrderedDict()
    code_to_card = OrderedDict()

    card_to_code['UNK'] = 0
    code_to_card[0] = 'UNK'

    with open(path_counts, 'rb') as f:
        cards_and_counts = pickle.load(f)

    sorted_by_count = sorted(cards_and_counts.items(),
                             key=lambda t: t[1],
                             reverse=True)

    for i, item in enumerate(sorted_by_count):
        name, _ = item
        if n != 'all' and i >= n:
            break

        card_to_code[name] = i + 1
        code_to_card[i + 1] = name

    with open(os.path.join(out_dir, 'encoding' + suffix), 'wb+') as f:
        pickle.dump(card_to_code, f)

    with open(os.path.join(out_dir, 'decoding' + suffix), 'wb+') as f:
        pickle.dump(code_to_card, f)


if __name__ == '__main__':
    make_code_mapping()
