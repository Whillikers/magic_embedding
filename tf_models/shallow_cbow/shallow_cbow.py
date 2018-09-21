'''
Model class for a 1-layer "conditional bag of words" embedding model,
which predicts a single card from all the other cards in a deck.
'''

import tensorflow as tf
from tf_models.model import TFModelABC


class ShallowCBOW(TFModelABC):
    model_args = ['embedding_size', 'n_negative_samples']

    def __init__(self, config):
        super().__init__(config, self.model_args)

    def build_graph(self):
        # Initialization scheme taken from
        # github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py
        init_width = 0.5 / self.config['embedding_size']
        self.embeddings = tf.Variable(
            tf.random_uniform([self.config['n_cards'] + 1,
                               self.config['embedding_size']],
                              minval=-init_width, maxval=init_width,
                              name='card_embedding_initializer'),
            name='card_embedding')
        tf.summary.histogram('card_embeddings', self.embeddings)

        with tf.name_scope('nce_vars'):
            self.nce_weights = tf.Variable(
                tf.zeros([self.config['n_cards'] + 1,
                          self.config['embedding_size']]), name='weights')
            tf.summary.histogram('nce_weights', self.nce_weights)

            self.nce_biases = tf.Variable(
                tf.zeros([self.config['n_cards'] + 1], name='biases'))
            tf.summary.histogram('nce_biases', self.nce_biases)

        # Graph nodes for inference
        with tf.name_scope('inference'):
            self.k = tf.placeholder(dtype='int32', shape=[], name='k')

            with tf.name_scope('nearest_cards'):
                self.input_id = tf.placeholder(dtype='int32', shape=[1],
                                               name='input_id')
                input_embedding = tf.nn.embedding_lookup(
                    self.embeddings, self.input_id, name='input_embedding')

                # Cosine similarities of input embeddings and all embeddings
                # Shape: [n_cards + 1]
                similarities = tf.matmul(input_embedding, self.embeddings,
                                         transpose_b=True, name='similarities')
                self.neighbor_similarities, self.neighbor_ids = \
                    tf.nn.top_k(similarities, self.k, name='similar_ids')

            with tf.name_scope('arithmetic'):
                pass  # TODO

    def get_train_ops(self, iterator):
        # NOTE: first dimension is batch index
        # context_ids has shape: [batch_size, n_context_cards]
        with tf.name_scope('inputs'):
            target_id, target_count, context_ids, context_counts = \
                iterator.get_next()

        with tf.name_scope('compute_embeddings'):
            # Map each card in the context to its embedding vector
            # Shape: [batch_size, n_context_cards, embedding_size]
            context_embeddings = tf.nn.embedding_lookup(
                self.embeddings, context_ids, name='context_embeddings')

            # TODO: change this to working at graph construction time
            # Create either weights or filters for non-present cards
            # Shape: [batch_size, n_context_cards]
            #  def dup_weights(): return tf.cast(context_counts, tf.float32)  # NOQA
            #  def no_dup_weights(): return tf.cast(tf.greater(context_counts, 0),  # NOQA
            #                                       tf.float32)
            #  context_weights = tf.cond(tf.cast(self.config['do_dup'], tf.bool),
            #                            dup_weights, no_dup_weights,
            #                            name='context_weights')

            context_weights = tf.cast(tf.greater(context_counts, 0),
                                                 tf.float32)

            weights_tiled = tf.tile(tf.expand_dims(context_weights, axis=2),
                                    (1, 1, self.config['embedding_size']))

            context_embeddings_weighted = context_embeddings * weights_tiled

            # Average embeddings across the context
            # Shape: [batch_size, embedding_size]
            sum_context_embedding = tf.reduce_sum(
                context_embeddings_weighted, axis=1,
                name='sum_context_embedding')

            tot_weights = tf.reduce_mean(weights_tiled, axis=1,
                                         name='tot_weights')

            mean_context_embedding = tf.divide(
                sum_context_embedding,
                tot_weights,
                name='mean_context_embedding')

        with tf.name_scope('compute_loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weights, biases=self.nce_biases,
                    labels=tf.expand_dims(target_id, axis=-1),
                    inputs=mean_context_embedding,
                    num_sampled=self.config['n_negative_samples'],
                    num_classes=self.config['n_cards'] + 1),
                name='batch_loss')

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(self.config['lr'])
            train = optimizer.minimize(loss, global_step=self.global_step)

        return train, loss

    def k_nearest_cards(self, card_id, k=10):
        out = self.sess.run([self.neighbor_ids],
                            feed_dict={self.k: k, self.input_id: [card_id]})
        return out[0][0]

    def k_nearest_arithmetic(self, add_ids, sub_ids, k=10):
        pass  # TODO
