import os
import pickle
from glob import glob
import click


@click.command()
@click.option('--out', default='datasets/counts/cards_and_counts.pkl')
@click.option('--dup', default=False, help='count duplicate cards in a deck')
@click.option('--sideboard', default=False, help='count cards in sideboards')
@click.option('--verbose', default=True)
@click.argument('dir_decks', default='datasets/raw_decks/deckbox')
def count_cards(out, dup, sideboard, verbose, dir_decks):
    '''
    Count how often each card is used and store the result as a pickled dict
    of form {card_name: card_count}.
    '''
    paths = glob(os.path.join(dir_decks, '*_mainboard.txt'))
    if sideboard:
        paths.extend(glob(os.path.join(dir_decks, '*_sideboard.txt')))

    cards_and_counts = {}

    if verbose:
        print('Collecting counts from path {}'.format(dir_decks))

    for i, path in enumerate(paths):
        if verbose and i % 10000 == 0:
            print('On deck: {}'.format(i))

        with open(path, 'r') as deck:
            lines = deck.readlines()

        for line in lines:
            count, name = line.strip().split(' ', 1)
            count = int(count)
            if not dup:
                count = count > 0

            if name in cards_and_counts.keys():
                cards_and_counts[name] += 1
            else:
                cards_and_counts[name] = 1

    if verbose:
        print('Done collecting counts. Saving...')

    with open(out, 'wb+') as f:
        pickle.dump(cards_and_counts, f)

    if verbose:
        print('Done!')


if __name__ == '__main__':
    count_cards()
