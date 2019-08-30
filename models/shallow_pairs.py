import tensorflow as tf

from models.model_abc import EmbeddingModelABC


class ShallowPairEmbedding(EmbeddingModelABC):
    """
    An embedding model which learns by comparing single pairs of cards.

    For method documentation, see superclass.
    """

    def __init__(self, n_cards, embedding_size, out_dir, n_negative_samples=128):
        super().__init__(n_cards, embedding_size, out_dir)

        self._n_negative_samples = n_negative_samples

        self.card_embedding = tf.keras.layers.Embedding(
            n_cards + 1,
            embedding_size,
            embeddings_initializer="uniform",
            name="card_embedding",
        )
        self.nce_weights = tf.Variable(
            tf.zeros_initializer()(shape=[embedding_size, n_cards])
        )
        self.nce_biases = tf.Variable(tf.zeros_initializer()(shape=[n_cards]))

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape + [self.embedding_size])

    def call(self, card_ids):
        return self.card_embedding(card_ids)

    def _k_nearest_ids_to_embeddings(self, embeddings, k=10):
        # Shape: [n_input_embeddings, n_cards]
        similarities = tf.matmul(
            embeddings,
            self.card_embedding.embeddings,
            transpose_b=True,
            name="similarity_matrix",
        )
        top_similarities, top_indices = tf.math.top_k(similarities, k)
        return top_indices, top_similarities

    def _loss(self, batch):
        """
        Compute the loss given a context, target pair.
        """
        target, context = batch
        embeddings = self.call(target)

        return tf.nn.nce_loss(
            weights=self.prediction_head.weights,
            biases=self.prediction_head.bias,
            labels=tf.expand_dims(context, axis=-1),
            inputs=embeddings,
            num_sampled=self._n_negative_samples,
            num_classes=self.n_cards + 1,
            num_true=1,
        )
