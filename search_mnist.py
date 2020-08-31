import sys
from pprint import pprint
import random
import numpy as np
import tensorflow as tf
import optuna
from optuna.integration.tfkeras import TFKerasPruningCallback

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Dropout
from tensorflow.keras.utils import to_categorical

num_classes = 10 
image_size = 784

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_data = training_images.reshape(training_images.shape[0], image_size) 
test_data = test_images.reshape(test_images.shape[0], image_size)

training_labels = to_categorical(training_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

def objective(trial):
  pruner = TFKerasPruningCallback(trial, 'loss')

  model = Sequential()

  model.add(Input(shape=(image_size,)))

  n_layers = trial.suggest_categorical('layers', [1, 2, 3, 4])
  alpha = trial.suggest_uniform('alpha', 0.001, 0.05)
  dropout_rate = trial.suggest_uniform('dropout', 0.0, 0.3)
  dropout_start = trial.suggest_int('start', 0, n_layers)

  for index in range(n_layers):
    if index > dropout_start:
      model.add(Dropout(rate=dropout_rate))
    n_units = trial.suggest_categorical('layer_%d' % index, [16, 32, 64, 128, 256])
    model.add(Dense(units=n_units))
    model.add(LeakyReLU(alpha=alpha))

  model.add(Dense(units=num_classes, activation='softmax'))

  model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
  history = model.fit(training_data, training_labels,
      batch_size=128,
      epochs=15,
      verbose=0,
      validation_split=.2,
      shuffle=True,
      callbacks=[pruner])
  print('metrics: {:.4f} {:.4f} {:.4f} {:.4f}'.format(history.history['accuracy'][-1],
    history.history['loss'][-1],
    history.history['val_accuracy'][-1],
    history.history['val_loss'][-1]), file=sys.stderr)
  sys.stderr.flush()
  return history.history['accuracy'][-1]


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)