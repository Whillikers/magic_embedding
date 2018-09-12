import sys
import implicit
from scipy.sparse import load_npz
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from matrix_models import utils

path_csr = 'datasets/matrix/card_deck_csr_5000.npz'
matrix = load_npz(path_csr)
train, val, deck_ids = utils.train_val_split(matrix)

run = 0


def objective(args):
    factors = int(args['factors'])
    regularization = args['regularization']
    iterations = int(args['iterations'])

    global run
    run += 1
    print('------------------------------------')
    print('Run:', run)
    print('Factors:', factors)
    print('Regularization:', regularization)
    print('Iterations:', iterations)
    sys.stdout.flush()

    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 regularization=regularization,
                                                 iterations=iterations)
    model.fit(train)
    val_acc = utils.top_n_frac(model, train, val, deck_ids)
    print('Acc:', val_acc)
    sys.stdout.flush()
    return {'loss': -1 * val_acc, 'status': STATUS_OK}


space = {
    'factors': hp.uniform('factors', 4, 512),
    'regularization': hp.uniform('regularization', 0, 0.5),
    'iterations': hp.uniform('iterations', 15, 50)
}

trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

print('best:', best)
print('trials:')
for trial in trials.trials[:2]:
    print(trial)
sys.stdout.flush()
