'''
Train a dummy model.
'''

from tf_models.dummy_model.dummy_model import DummyModel

config = {
    'model_name': 'dummy',
    'save_dir': 'logs/dummy/',
    'n_decks': 100,
    'n_cards': 5000,
    'do_sideboard': False,
    'do_dup': False,
    'batch_size': 8,
    'val_frac': 0.2,
    'lr': 1e-3,
    'max_epochs': 10,
    'summary_interval': 2,
    'save_interval': 2,
    'print_cards': ['Brainstorm', 'Lightning Bolt', 'Doom Blade',
                    'Forest', 'Goblin Guide'],
    'random_seed': 1337,
    'debug_mode': False
}

model = DummyModel(config)
model.train()
