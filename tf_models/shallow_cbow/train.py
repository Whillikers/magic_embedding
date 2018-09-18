'''
Train a shallow CBOW model.
'''

from tf_models.shallow_cbow.shallow_cbow import ShallowCBOW

config = {
    'embedding_size': 64,
    'n_negative_samples': 256,  # TODO: tune
    'model_name': 'shallow_cbow',
    'save_dir': 'logs/shallow_cbow/',
    'n_decks': 580400,
    'n_cards': 1000,
    'do_sideboard': False,
    'do_dup': False,
    'batch_size': 32,  # TODO: tune
    'val_frac': 0.2,
    'lr': 1e-3,  # TODO: tune
    'max_epochs': 10,  # TODO
    'summary_interval': 10,  # TODO
    'save_interval': 1,  # TODO
    'print_cards': ['Brainstorm', 'Lightning Bolt', 'Doom Blade',
                    'Forest', 'Goblin Guide'],
    'log_level': 20,
    'random_seed': 1337
}

if __name__ == '__main__':
    model = ShallowCBOW(config)
    ds = model.get_dataset()
    model.train(ds)
