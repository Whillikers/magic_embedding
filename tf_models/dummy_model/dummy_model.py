import tensorflow as tf
from tf_models.models.model import TFModelABC


class DummyModel(TFModelABC):
    '''
    Dummy Tensorflow thesaurus model to test core functions.
    '''

    def __init__(self, config):
        super().__init__(config)

    def build_graph(self):
        pass

    def get_train_ops(self, iterator):
        with tf.name_scope('training'):
            next_ex = iterator.get_next(name='input')
            inc_global_step = tf.assign_add(self.global_step, 1)
            with tf.control_dependencies([next_ex[0], inc_global_step]):
                train_op = tf.no_op(name='dummy_train_op')
                loss_many = tf.random_normal([10], name='loss_many')
                tf.summary.histogram('loss_hist', loss_many)
                loss_op = tf.reduce_sum(loss_many, name='loss')
                return train_op, loss_op

    def k_nearest_cards(self, card_id, k=10):
        return list(range(card_id, card_id + k))
