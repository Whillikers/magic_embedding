import os
from functools import partial
import pickle
from glob import glob
import click
import numpy as np
import tensorflow as tf


def get_card_id_and_count(line, encoding):
    count, name = line.strip().split(' ', 1)
    try:
        code = encoding[name]
    except KeyError:
        code = encoding['UNK']
    return (code, int(count))


def list_to_feature(lst):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(lst)))


@click.command()
@click.option('--n', default='all')
@click.option('--out_dir', default='datasets/tfrecord_decks/')
@click.option('--decks_dir', default='datasets/raw_decks/deckbox')
def make_tfrecord(n, out_dir, decks_dir):
    print('Making TFRecord dataset from dir:', decks_dir)
    source_name = decks_dir.split('/')[-1]
    out_path = os.path.join(out_dir, source_name + '_' + n + '.tfrecord')
    writer = tf.python_io.TFRecordWriter(out_path)
    mainboard_paths = glob(os.path.join(decks_dir, '*_mainboard.txt'))
    print('Decks:', len(mainboard_paths))
    with open('datasets/code_mappings/encoding_' + n + '.pkl', 'rb') as f:
        encoding = pickle.load(f)
    for i, mainboard_path in enumerate(mainboard_paths):
        if i % 10000 == 0:
            print('On deck:', i)
        sideboard_path = mainboard_path.replace('_mainboard', '_sideboard')
        with open(mainboard_path, 'r') as f:
            mainboard_lines = f.readlines()
        with open(sideboard_path, 'r') as f:
            sideboard_lines = f.readlines()

        map_fn = partial(get_card_id_and_count, encoding=encoding)

        mainboard_ids_and_counts = list(map(map_fn, mainboard_lines))
        mainboard_ids, mainboard_counts = zip(*mainboard_ids_and_counts)

        if sideboard_lines:
            sideboard_ids_and_counts = list(map(map_fn, sideboard_lines))
            sideboard_ids, sideboard_counts = zip(*sideboard_ids_and_counts)
        else:
            sideboard_ids = []
            sideboard_counts = []

        feature = {
            'mainboard_ids': list_to_feature(mainboard_ids),
            'mainboard_counts': list_to_feature(mainboard_counts),
            'sideboard_ids': list_to_feature(sideboard_ids),
            'sideboard_counts': list_to_feature(sideboard_counts)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


if __name__ == '__main__':
    make_tfrecord()
