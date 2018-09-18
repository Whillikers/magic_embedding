'''
Utilities related to making and working with datasets.
'''

from functools import partial
import tensorflow as tf


# Top-level functions #
def get_dataset_singleton(path='datasets/tfrecord_decks/deckbox_5000.tfrecord',
                          do_sideboard=True, do_dup=True,
                          batch_size=128, random_seed=1337):
    return make_dataset_singleton(get_dataset_decks(path, do_sideboard,
                                                    do_dup))\
        .shuffle(100000, seed=random_seed)\
        .padded_batch(batch_size, ([], [], [None], [None]))


def get_dataset_decks(path='datasets/tfrecord_decks/deckbox_5000.tfrecord',
                      do_sideboard=True, do_dup=True):
    feature_set = {
        'mainboard_ids': tf.VarLenFeature(tf.int64),
        'mainboard_counts': tf.VarLenFeature(tf.int64),
        'sideboard_ids': tf.VarLenFeature(tf.int64),
        'sideboard_counts': tf.VarLenFeature(tf.int64)
    }

    def get_cards_mainboard(example):
        ids = tf.sparse_tensor_to_dense(example['mainboard_ids'])
        counts = tf.sparse_tensor_to_dense(example['mainboard_counts'])
        return ids, counts

    def get_cards_both(example):
        ids_main = tf.sparse_tensor_to_dense(example['mainboard_ids'])
        counts_main = tf.sparse_tensor_to_dense(example['mainboard_counts'])
        ids_side = tf.sparse_tensor_to_dense(example['sideboard_ids'])
        counts_side = tf.sparse_tensor_to_dense(example['sideboard_counts'])

        ids = tf.concat(values=[ids_main, ids_side], axis=0)
        counts = tf.concat(values=[counts_main, counts_side], axis=0)
        return ids, counts

    def collapse_unq(ids, counts):
        '''
        Resolve duplicates ignoring counts, treats all counts as 1.
        '''
        unq_ids, _ = tf.unique(ids)
        return unq_ids, tf.ones_like(unq_ids)

    def collapse_dup(ids, counts):
        '''
        Sum the counts of duplicate cards. Used to resolve multiple UNK tokens
        and mainboard/sideboard duplication.
        '''
        unq_ids, old_idx = tf.unique(ids)
        n_segments = tf.reduce_max(old_idx) + 1
        merged_counts = tf.unsorted_segment_sum(data=counts,
                                                segment_ids=old_idx,
                                                num_segments=n_segments)
        return unq_ids, merged_counts

    parse_example = partial(tf.parse_single_example, features=feature_set)
    card_merger = get_cards_both if do_sideboard else get_cards_mainboard
    dataset = tf.data.TFRecordDataset([path]) \
        .map(parse_example) \
        .map(card_merger)
    return dataset.map(collapse_dup) if do_dup else dataset.map(collapse_unq)


# Utility functions #
def make_dataset_singleton(dataset):
    '''
    Given a dataset of (list of card ids, list of card counts), for each
    example, create a list of examples with one card/count pair "pulled out."

    Output dataset format: (
        single card id,
        single card count,
        list of card ids,
        list of card counts
    )
    '''
    def make_examples_singleton(ids, counts):
        n_cards = tf.size(ids)
        card_idxs = tf.range(start=0, limit=n_cards)

        def make_one_example(card_idx):
            mask_rest = tf.not_equal(card_idxs, card_idx)
            single_id = ids[card_idx]
            single_count = counts[card_idx]
            rest_ids = tf.boolean_mask(ids, mask_rest)
            rest_counts = tf.boolean_mask(counts, mask_rest)
            return single_id, single_count, rest_ids, rest_counts
        outs = tf.map_fn(make_one_example, card_idxs, back_prop=False,
                         dtype=(tf.int64, tf.int64, tf.int64, tf.int64))
        return tf.data.Dataset.from_tensor_slices(outs)

    return dataset.flat_map(make_examples_singleton)
