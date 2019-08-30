'''
Train a shallow CBOW model.
'''

from tf_models.shallow_cbow.shallow_cbow import ShallowCBOW

config = {
    'embedding_size': 64,
    'n_negative_samples': 256,  # TODO: tune
    'model_name': 'shallow_cbow',
    'save_dir': 'logs/shallow_cbow/',
    #  'n_decks': 580400,
    'n_decks': 100,
    'n_cards': 1000,
    'do_sideboard': False,
    'do_dup': False,
    'batch_size': 32,  # TODO: tune
    'val_frac': 0.2,
    'lr': 1e-2,  # TODO: tune
    'max_epochs': 100,  # TODO
    'summary_interval': 1,
    'save_interval': 1,
    'print_cards': ['Brainstorm', 'Lightning Bolt', 'Doom Blade',
                    'Forest', 'Goblin Guide'],
    'random_seed': 1337,
    'debug_mode': False
}

if __name__ == '__main__':
    model = ShallowCBOW(config)
    with model.graph.as_default():
        dataset = model.get_dataset()
    model.train(dataset)
