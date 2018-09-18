'''
ABC for Tensorflow thesaurus models.

Based off structure suggested by:
blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-
and-models-architecture-f23171501ae3
'''

import os
import pickle
import logging
from copy import deepcopy
from abc import ABC, abstractmethod
import tensorflow as tf
from tf_models.utils.dataset import get_dataset_singleton


class TFModelABC(ABC):
    '''
    ABC for Tensorflow thesaurus models.
    '''
    global_args = [
        'model_name', 'save_dir',                               # model
        'n_decks', 'n_cards', 'do_sideboard', 'do_dup',         # data
        'batch_size', 'val_frac', 'lr', 'max_epochs',           # training
        'summary_interval', 'save_interval', 'print_cards',     # misc
        'log_level', 'random_seed'
    ]

    def __init__(self, config, model_args=[]):
        '''
        Initialize a TFModelABC.

        model_args is a list of strings specifying the names of required
        arguments specific to the model.
        '''

        # Validate and load args
        for arg_name in self.global_args + model_args:
            assert(arg_name in config.keys())

        self.config = deepcopy(config)

        # Set up logging
        self.checkpoint_dir = os.path.join(config['save_dir'], 'checkpoints/')
        self.tf_log_dir = os.path.join(config['save_dir'], 'tflogs/')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tf_log_dir, exist_ok=True)

        self.logger = logging.Logger(config['model_name'] + '_logger',
                                     level=logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s: %(message)s')
        log_fh = logging.FileHandler(os.path.join(config['save_dir'],
                                                  'logs.log'))
        log_fh.setLevel(config['log_level'])
        log_fh.setFormatter(formatter)
        debug_fh = logging.FileHandler(os.path.join(config['save_dir'],
                                                    'debug.log'))
        debug_fh.setLevel(logging.DEBUG)
        debug_fh.setFormatter(formatter)
        print_fh = logging.StreamHandler()
        print_fh.setFormatter(formatter)
        print_fh.setLevel(config['log_level'])
        self.logger.addHandler(debug_fh)
        self.logger.addHandler(log_fh)
        self.logger.addHandler(print_fh)

        self.logger.debug('loading card ID mappings')
        map_base = 'datasets/code_mappings/{}_{}.pkl'.format('{}',
                                                             config['n_cards'])
        with open(map_base.format('encoding'), 'rb') as f:
            self.name_to_id = pickle.load(f)
        with open(map_base.format('decoding'), 'rb') as f:
            self.id_to_name = pickle.load(f)

        self.logger.debug('configuring session and base graph')
        self.graph = tf.Graph()
        tf.set_random_seed(config['random_seed'])

        session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions()
        )

        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph, config=session_config)

            self.logger.debug('building base ops')
            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)

            # TODO: remove when done
            with tf.name_scope('debug'):
                self.debug_feed = tf.placeholder(tf.int32, [], 'debug_feed')

            self.logger.debug('building model graph')
            self.build_graph()

            self.logger.debug('initializing all variables')
            self.sess.run(tf.global_variables_initializer())

            try:
                self.init()
                self.logger.debug('running model-defined init()')
            except NotImplementedError:
                self.logger.warning('no model-defined init() found')

    # Top-level methods #
    def save(self, step=None):
        with self.graph.as_default():
            savepath = os.path.join(self.checkpoint_dir,
                                    self.config['model_name'] + '.ckpt')
            saved_path = self.saver.save(self.sess, savepath, global_step=step)
            self.logger.info('saved checkpoint at path: {}'.format(saved_path))

    def load(self, path=None):
        with self.graph.as_default():
            if not path:
                path = os.path.join(self.checkpoint_dir,
                                    self.config['model_name'] + '.ckpt')

            self.saver.restore(self.sess, path)
            self.logger.info('restored from checkpoint at path: {}'
                             .format(path))

    def train(self, dataset=None):
        '''
        Train a model, handling saving, metrics, gradient updates, etc.
        Assumes dataset is shuffled and batched, if appropriate.
        '''
        with self.graph.as_default():
            if not dataset:
                self.logger.warning('no dataset found; using default')
                dataset = self.get_dataset()

            self.logger.info('preparing datasets')
            n_val = int(self.config['val_frac'] * self.config['n_decks'])

            with tf.name_scope('data'):
                dataset_train = dataset.skip(n_val)\
                    .take(self.config['n_decks'] - n_val).cache()
                dataset_val = dataset.take(n_val).cache()

                self.logger.info('{} training decks, {} validation decks'
                                 .format(self.config['n_decks'] - n_val,
                                         n_val))

                iterator = tf.data.Iterator.from_structure(
                    dataset.output_types, dataset.output_shapes)
                train_init_op = iterator.make_initializer(
                    dataset_train, name='initialize_train_iterator')
                val_init_op = iterator.make_initializer(
                    dataset_val, name='initialize_val_iterator')

            self.logger.debug('building training graph')
            train_op, loss_op = self.get_train_ops(iterator)

            with tf.name_scope('visualization'):
                mean_loss, mean_loss_update = tf.metrics.mean(
                    loss_op, name='mean_loss')
                tf.summary.scalar('mean_loss', mean_loss)
                summary_op = tf.summary.merge_all()

            self.logger.debug('creating saver and file writers')
            self.saver = tf.train.Saver()
            self.train_writer = tf.summary.FileWriter(
                os.path.join(self.tf_log_dir, 'train/'), graph=self.sess.graph)
            self.val_writer = tf.summary.FileWriter(
                os.path.join(self.tf_log_dir, 'val/'), graph=self.sess.graph)

            self.logger.info('beginning training')
            for epoch in range(self.config['max_epochs']):
                self.logger.info('epoch: {}'.format(epoch))

                # Train and val step, with extra debugging info
                if self.config['summary_interval'] and \
                        (epoch % self.config['summary_interval'] == 0):
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)

                    self.logger.debug('initializing run metadata for train')
                    run_metadata = tf.RunMetadata()

                    self.logger.debug('initializing local variables for train')
                    self.sess.run(tf.local_variables_initializer(),
                                  options=run_options,
                                  run_metadata=run_metadata)
                    self.sess.run(train_init_op, options=run_options,
                                  run_metadata=run_metadata)

                    self.logger.info('running training with metadata')
                    while True:
                        try:
                            _, mean_train_loss, step, train_summ = \
                                self.sess.run([train_op, mean_loss_update,
                                               self.global_step, summary_op],
                                              options=run_options,
                                              run_metadata=run_metadata,
                                              feed_dict={self.debug_feed: 1})
                        except tf.errors.OutOfRangeError:
                            break

                    self.logger.info('training loss: {}'.format(
                        mean_train_loss))

                    self.logger.info('writing train summaries')
                    self.train_writer.add_run_metadata(run_metadata,
                                                       'step_{}'.format(step))
                    self.train_writer.add_summary(train_summ, step)

                    self.logger.debug('initializing run metadata for val')
                    run_metadata = tf.RunMetadata()

                    self.logger.debug('initializing local variables for val')
                    self.sess.run(tf.local_variables_initializer())
                    self.sess.run(val_init_op)

                    self.logger.info('running validation')
                    while True:
                        try:
                            mean_val_loss, val_summ = self.sess.run(
                                [mean_loss_update, summary_op],
                                options=run_options, run_metadata=run_metadata,
                                feed_dict={self.debug_feed: 2})

                        except tf.errors.OutOfRangeError:
                            break

                    self.logger.info('val loss: {}'.format(mean_val_loss))

                    self.val_writer.add_run_metadata(run_metadata,
                                                     'step_{}'.format(step))
                    self.val_writer.add_summary(val_summ, step)

                    # Periodically print some card associations
                    self.logger.debug('printing similar cards')
                    for card_name in self.config['print_cards']:
                        id_ = self.name_to_id[card_name]
                        closest_5_ids = self.k_nearest_cards(id_, 5)
                        closest_5_names = list(
                            map(lambda i: self.id_to_name[i], closest_5_ids))
                        self.logger.info('closest to {}:'.format(card_name))
                        self.logger.info(closest_5_names)

                else:  # Normal training step
                    self.logger.debug('initializing local variables for train')
                    self.sess.run(tf.local_variables_initializer())
                    self.sess.run(train_init_op)

                    self.logger.info('running training')
                    while True:
                        try:
                            _, mean_train_loss, step, summ = self.sess.run(
                                [train_op, mean_loss_update,
                                 self.global_step, summary_op],
                                feed_dict={self.debug_feed: 1})
                        except tf.errors.OutOfRangeError:
                            break
                    self.logger.info('training loss: {}'.format(
                        mean_train_loss))

                    self.train_writer.add_summary(train_summ, step)

                # Save model periodically
                if self.config['save_interval'] and \
                        (epoch % self.config['save_interval'] == 0):
                    self.save(step)

    # Required subclass methods #
    @abstractmethod
    def build_graph(self):
        '''
        Build all of the background operations required to train and apply
        this model.
        '''
        pass

    @abstractmethod
    def get_train_ops(self, iterator):
        '''
        Build the subgraph to get the 2-tuple of ops:
         - Optimizer loss minimization op
         - Training loss
        Each is computed over a single batch.
        '''
        pass

    @abstractmethod
    def k_nearest_cards(self, card_id, k=10):
        '''
        Return the k card IDs that are most similar to card_id.
        '''
        pass

    # Optional subclass methods #
    def init(self):
        '''
        Optional extra init for a model.
        '''
        raise NotImplementedError('{} has no init step'.format(self.__class__))

    def k_nearest_arithmetic(self, add_ids, sub_ids, k=10):
        '''
        Return the k card IDs most similar to sum(add_ids) - sum(sub_ids).

        Optional; a subclass can implement if it supports embedding arithmetic.
        '''
        raise NotImplementedError(
            '{} does not support card arithmetic'.format(self.__class__))

    def get_dataset(self):
        '''
        Get the dataset associated to this model's parameters.
        Has a default value but can be overwritten by a subclass.
        '''
        path = 'datasets/tfrecord_decks/deckbox_{}.tfrecord'.format(
            self.config['n_cards'])
        return get_dataset_singleton(path,
                                     self.config['do_sideboard'],
                                     self.config['do_dup'],
                                     self.config['batch_size'],
                                     self.config['random_seed'])
