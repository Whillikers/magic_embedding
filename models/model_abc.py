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

    def k_nearest_cards(self, cards, k=10):
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
        """
        if type(cards[0]) == str:
            embeddings = self.card_embeddings(cards)
        else:
            embeddings = tf.constant(cards)

        similar_ids = self._k_nearest_ids_to_embeddings(embeddings, k)
        return [[self._decoding(id_) for id_ in row] for row in similar_ids]

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
    def _loss(self, batch):
        """
        Compute the loss associated with a set of features.

        Parameters
        ----------
        batch
            Some structured tf.Tensors, with exact structure depending on the
            dataset.

        Returns
        -------
        tf.Tensor
            Scalar representing the batch loss
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError  # TODO
