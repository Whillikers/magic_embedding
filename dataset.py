"""
Utilities related to making and working with datasets.
"""

from functools import partial

import tensorflow as tf


# Top-level functions #
def get_all_pairs_dataset(path, batch_size, use_sideboard=False):
    """
    Get a dataset of all pairs of cards in all decks.

    Parameters
    ----------
    path : str path to tfrecord
        Path to tfrecord containing decks.
    batch_size : int
        Batch size to use.
    use_sideboard : bool
        Whether to include sideboard cards.

    Returns
    -------
    tf.data.Dataset
        The dataset.
    """
    ds_decks = get_dataset_decks(path, use_sideboard, do_dup=False).cache()

    @tf.function()
    def make_pairs(deck):
        n_cards = tf.reduce_sum(deck["counts"])
        ids = deck["ids"]

        # See:
        # https://stackoverflow.com/questions/47132665/cartesian-product-in-tensorflow
        tile_a = tf.tile(tf.expand_dims(ids, 1), [1, n_cards])
        tile_a = tf.expand_dims(tile_a, 2)
        tile_b = tf.tile(tf.expand_dims(ids, 0), [n_cards, 1])
        tile_b = tf.expand_dims(tile_b, 2)

        cartesian_product = tf.concat([tile_a, tile_b], axis=2)
        cartesian_product = tf.reshape(cartesian_product, [-1, 2])

        return tf.data.Dataset.from_tensor_slices(cartesian_product)

    return (
        ds_decks.flat_map(make_pairs)
        .shuffle(int(1e6))
        .batch(batch_size, drop_remainder=True)
    )


def get_dataset_decks(
    path="datasets/tfrecord_decks/deckbox_5000.tfrecord",
    use_sideboard=True,
    do_dup=True,
):
    feature_set = {
        "mainboard_ids": tf.io.VarLenFeature(tf.int64),
        "mainboard_counts": tf.io.VarLenFeature(tf.int64),
        "sideboard_ids": tf.io.VarLenFeature(tf.int64),
        "sideboard_counts": tf.io.VarLenFeature(tf.int64),
    }

    def get_cards_mainboard(example):
        ids = tf.sparse.to_dense(example["mainboard_ids"])
        counts = tf.sparse.to_dense(example["mainboard_counts"])
        return ids, counts

    def get_cards_both(example):
        ids_main = tf.sparse.to_dense(example["mainboard_ids"])
        counts_main = tf.sparse.to_dense(example["mainboard_counts"])
        ids_side = tf.sparse.to_dense(example["sideboard_ids"])
        counts_side = tf.sparse.to_dense(example["sideboard_counts"])

        ids = tf.concat(values=[ids_main, ids_side], axis=0)
        counts = tf.concat(values=[counts_main, counts_side], axis=0)
        return ids, counts

    def collapse_uniq(ids, counts):
        """
        Resolve duplicates ignoring counts, treats all counts as 1.
        """
        unq_ids, _ = tf.unique(ids)
        return unq_ids, tf.ones_like(unq_ids)

    def collapse_dup(ids, counts):
        """
        Sum the counts of duplicate cards. Used to resolve multiple UNK tokens
        and mainboard/sideboard duplication.
        """
        unq_ids, old_idx = tf.unique(ids)
        n_segments = tf.reduce_max(old_idx) + 1
        merged_counts = tf.math.unsorted_segment_sum(
            data=counts, segment_ids=old_idx, num_segments=n_segments
        )
        return unq_ids, merged_counts

    def as_dict(ids, counts):
        """
        Map the input into a nicer dictionary format.
        """
        return {"ids": ids, "counts": counts}

    parse_example = partial(tf.io.parse_single_example, features=feature_set)
    card_merger = get_cards_both if use_sideboard else get_cards_mainboard
    collapse_fn = collapse_dup if do_dup else collapse_uniq

    return (
        tf.data.TFRecordDataset([path])
        .map(parse_example)
        .map(card_merger)
        .map(collapse_fn)
        .map(as_dict)
    )


# Internals #
def _make_dataset_singleton(dataset):
    """
    Given a dataset of (list of card ids, list of card counts), for each
    example, create a list of examples with one card/count pair "pulled out."

    Output dataset format: {
        'single',
        'single_count',
        'context',
        'context_counts'
    }
    """

    def make_examples_singleton(features):
        ids = features["ids"]
        counts = features["counts"]
        n_cards = tf.size(ids)
        card_idxs = tf.range(start=0, limit=n_cards)

        def make_one_example(card_idx):
            mask_rest = tf.not_equal(card_idxs, card_idx)
            single_id = ids[card_idx]
            single_count = counts[card_idx]
            rest_ids = tf.boolean_mask(ids, mask_rest)
            rest_counts = tf.boolean_mask(counts, mask_rest)
            return {
                "single": single_id,
                "single_count": single_count,
                "context": rest_ids,
                "context_counts": rest_counts,
            }

        outs = tf.map_fn(
            make_one_example,
            card_idxs,
            back_prop=False,
            dtype={
                "single": tf.int64,
                "single_count": tf.int64,
                "context": tf.int64,
                "context_counts": tf.int64,
            },
        )
        return tf.data.Dataset.from_tensor_slices(outs)

    def is_example_valid(features):
        return tf.reduce_any(features["context_counts"] > 0)

    return dataset.flat_map(make_examples_singleton).filter(is_example_valid)
