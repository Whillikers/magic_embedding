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
            tf.zeros_initializer()(shape=[n_cards + 1, embedding_size])
        )
        self.nce_biases = tf.Variable(tf.zeros_initializer()(shape=[n_cards + 1]))

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape + [self.embedding_size])

    def call(self, card_ids):
        return self.card_embedding(card_ids)

    def _k_nearest_ids_to_embeddings(self, embeddings, k=10):
        with tf.name_scope("Nearest_Embeddings"):
            normalized_input = tf.math.l2_normalize(
                embeddings, axis=1, name="normalized_input_embeddings"
            )
            normalized_embeddings = tf.math.l2_normalize(
                self.card_embedding.embeddings,
                axis=1,
                name="normalized_card_embeddings",
            )

            # Shape: [n_input_embeddings, n_cards]
            similarities = tf.matmul(
                normalized_input,
                normalized_embeddings,
                transpose_b=True,
                name="similarity_matrix",
            )
            top_similarities, top_indices = tf.math.top_k(similarities, k)
            return top_indices, top_similarities

    def _loss(self, batch, step=None):
        targets = batch[:, 0]
        contexts = batch[:, 1]

        embeddings = self.call(targets)

        with tf.name_scope("Loss"):
            element_loss = tf.nn.nce_loss(
                weights=self.nce_weights,
                biases=self.nce_biases,
                labels=tf.expand_dims(contexts, axis=-1),
                inputs=embeddings,
                num_sampled=self._n_negative_samples,
                num_classes=self.n_cards + 1,
                num_true=1,
                name="element_loss",
            )
            batch_loss = tf.reduce_mean(element_loss, name="batch_loss")

        if step is not None:
            tf.summary.histogram("card_embeddings", embeddings, step)
            tf.summary.histogram("element_losses", element_loss, step)
            tf.summary.scalar("loss", batch_loss, step)

        return batch_loss
