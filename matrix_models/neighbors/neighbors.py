import implicit

name_to_model = {
    'cosine': implicit.nearest_neighbours.CosineRecommender,
    'tfidf': implicit.nearest_neighbours.TFIDFRecommender,
    'bm25': implicit.nearest_neighbours.BM25Recommender
}

kwargs_for_model = {
    'cosine': [],
    'tfidf': [],
    'bm25': ['K1, B']  # K1 ~ [1.2, 2], B = 0.75
}


def train_neighbors(train_matrix, k=10, model_type='cosine', **kwargs):
    model_class = name_to_model[model_type]
    model = model_class(K=k, **kwargs)
    model.fit(train_matrix)
    return model
