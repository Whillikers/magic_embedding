from functools import partial
import tensorflow as tf

path = 'datasets/tfrecord_decks/deckbox_1000.tfrecord'

feature_set = {
    'mainboard_ids': tf.VarLenFeature(tf.int64),
    'mainboard_counts': tf.VarLenFeature(tf.int64),
    'sideboard_ids': tf.VarLenFeature(tf.int64),
    'sideboard_counts': tf.VarLenFeature(tf.int64)
}

parse_example = partial(tf.parse_single_example, features=feature_set)

dataset = tf.data.TFRecordDataset([path]) \
    .map(parse_example)

iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()


def get_next():
    with tf.Session() as sess:
        return sess.run(example)
