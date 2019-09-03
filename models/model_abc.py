import pickle
from abc import ABC, abstractmethod

import tensorflow as tf


class EmbeddingModelABC(tf.keras.Model, ABC):
    def __init__(self, n_cards, embedding_size, out_dir):
        """
        Base class for all TF embedding models.

        Parameters
        ----------
        n_cards : int
            How many cards to store an embedding for, not including the UNK token.
        embedding_size : int
            Number of dimensions in the learned embedding space.
        out_dir : str
            Directory to place logs, checkpoints, etc.
        """
        super(EmbeddingModelABC, self).__init__(name="embedding_model")
        self.n_cards = n_cards
        self.embedding_size = embedding_size
        self.out_dir = out_dir

        with open(f"datasets/code_mappings/encoding_{n_cards}.pkl", "rb") as f:
            self._encoding = pickle.load(f)

        with open(f"datasets/code_mappings/decoding_{n_cards}.pkl", "rb") as f:
            self._decoding = pickle.load(f)

    def card_embeddings(self, card_names):
        """
        Compute the embeddings associated with a list of card names.

        Parameters
        ----------
        card_names : [string]
            Names of cards to retrieve embeddings for.

        Returns
        -------
        tf.Tensor of shape [len(card_ids), embedding_size]
            The embeddings.
        """
        card_ids = [self._encoding[name] for name in card_names]
        return self.call(card_ids)

    def k_nearest_cards_and_similarities(self, cards, k=10):
        """
        For each card in cards, return the k card names that are most similar.

        Parameters
        ----------
        card_names : [string] or [ndarray]
             Cards to retrieve neighbors for.
             If [string], acts as card names and must be keys in self._encoding.
             If [ndarray], acts as card embeddings and must have dimensions
             equal to self.embedding_size.
        k : [int]
            How many neighbors of each card to retrieve.

        Returns
        -------
        [[str]]
            For each card, a list of other card ids most similar.
        [[float]]
            For each card, how similar each other card is.
        """
        # TODO: solve tensor problem

        if type(cards[0]) == str:
            embeddings = self.card_embeddings(cards)
        else:
            embeddings = cards

        similar_ids, similarities = self._k_nearest_ids_to_embeddings(embeddings, k)
        similar_cards = [
            [self._decoding[id_] for id_ in row] for row in similar_ids.numpy()
        ]
        return similar_cards, similarities

    @abstractmethod
    def call(self, card_ids):
        """
        Compute the embeddings associated with a list of card IDs.

        Parameters
        ----------
        card_ids : [int]
            IDs of cards to retrieve embeddings for.

        Returns
        -------
        tf.Tensor of shape [len(card_ids), embedding_size]
            The embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def _k_nearest_ids_to_embeddings(self, embeddings, k=10):
        """
        For each embedding in embeddings, return the k card IDs that are most similar.

        Parameters
        ----------
        embeddings : tf.Tensor
            IDs of cards to retrieve similar cards for.
        k : [int]
            How many neighbors of each card to retrieve.

        Returns
        -------
        tf.Tensor of shape [len(embeddings), k]
            k card IDs that are most similar to the input embeddings.
        tf.Tensor of shape [len(embeddings), k]
            Similarities of those card IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def _loss(self, batch, step=None):
        """
        Compute the loss associated with a set of features.

        Parameters
        ----------
        batch
            Some structured tf.Tensors, with exact structure depending on the
            dataset.
        step (optional) : None or int
            The step used for plotting summaries.
            If None, do not make summaries.

        Returns
        -------
        tf.Tensor
            Scalar representing the batch loss
        """
        raise NotImplementedError

    def train(
        self,
        dataset,
        learning_rate,
        watch_cards=["Lightning Bolt", "Brainstorm", "Command Tower", "Thoughtseize"],
    ):
        """
        TODO
        """

        optimizer = tf.optimizers.Adam(learning_rate, clipnorm=1)
        writer = tf.summary.create_file_writer(self.out_dir)
        watch_ids = tf.constant([self._encoding[name] for name in watch_cards])

        # TODO: checkpoints
        # TODO: embedding projector
        with writer.as_default():
            for step, batch in enumerate(dataset):
                with tf.GradientTape() as tape:
                    loss = self._loss(batch, step)

                gradients = tape.gradient(loss, self.trainable_variables)
                grads_and_vars = list(zip(gradients, self.trainable_variables))
                for grad, var in grads_and_vars:
                    tf.summary.histogram(
                        "gradients/{}".format(var.name), grad.values, step
                    )
                optimizer.apply_gradients(grads_and_vars)

                # For each of a few chosen cards, show a table of other similar cards
                watch_embeddings = self.call(watch_ids)
                similar_cards, similarities = self.k_nearest_cards_and_similarities(
                    watch_embeddings
                )
                similar_card_table = self._format_similar_card_table(
                    watch_cards, similar_cards, similarities
                )
                tf.summary.histogram("watch_embeddings", watch_embeddings, step)
                tf.summary.text("watch_similar_cards", similar_card_table, step)

    @staticmethod
    def _format_similar_card_table(header_cards, similar_cards, similarities):
        """
        Format a table of cards and their neighbors in embedding space.

        See: https://github.com/tensorflow/tensorboard/blob/master/
             tensorboard/plugins/text/text_demo.py

        Parameters
        ----------
        header_cards : [string]
            The cards used as anchors.
        similar_cards : [[string]]
            For each anchor card, a list of similar cards.
        similarties: [[float]]
            For each card, how similar each other card is.

        Returns
        -------
        string
            A Markdown table of the input cards
        """
        header = "|".join(header_cards)
        subheader = "---|---" * (len(header_cards) - 1)

        sims_formatted = [
            [f"{card} ({sim.numpy():.3f})" for card, sim in zip(*pair_list)]
            for pair_list in zip(similar_cards, similarities)
        ]
        rows = ["|".join(list(row)) for row in zip(*sims_formatted)]
        table = "{}\n{}\n{}".format(header, subheader, "\n".join(rows))
        return table
