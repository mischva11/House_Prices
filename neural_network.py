import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.layers import Dropout


def batch_generator(batch_size, sequence_length, x_train_scaled, y_train_scaled, num_train, num_y_signals, num_x_signals):


    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]

        yield (x_batch, y_batch)



    # Maybe use lower init-ranges.

#def loss function
def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

init = RandomUniform(minval=-0.05, maxval=0.05)
fit = True
#def batch size
batch_size = 64
sequence_length = 100

#data setup
result_2 =[]

##data setup
nrow=10000

train = pd.read_csv("data/processed.csv", nrows=nrow)
df = train
#filter df
target_var = ["answered_correctly"]
used_var = ["answered_correctly", "content_id", "content_type_id"]#, "prior_question_elapsed_time", "prior_question_had_explanation"]
target_names = target_var
df = df[used_var]

#steps to predict
shift_steps = 5000
df_target = df[target_var].shift(-shift_steps)
##neural network daten setup
#data
x_data = df.values[0:-shift_steps]
print("Shape:", x_data.shape)
#target
y_data = df_target.values[:-shift_steps]
print("Shape:", y_data.shape)
#length of data
num_data = len(x_data)
#fraction of the data-set that will be used for the training-set
train_split = 0.9
#number of observations in the training-set
num_train = int(train_split * num_data)
#number of observations
num_test = num_data - num_train
#input signals
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
#output signals
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
#number of input signals
num_x_signals = x_data.shape[1]
#number of output signals
num_y_signals = y_data.shape[1]

#scaling
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.fit_transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

#batches
# del x_data
# del y_data
del df
generator = batch_generator(batch_size, sequence_length, x_train_scaled, y_train_scaled, num_train, num_y_signals, num_x_signals)

x_batch, y_batch = next(generator)

print(x_batch.shape)
print(y_batch.shape)

#set up validation set
validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


#gru modell - keras
model = Sequential()
dropout = 0.1
#LSTM modell

model.add(LSTM(units=128,
              return_sequences=True,
              input_shape=(None,num_x_signals,)))
model.add(Dropout(dropout))
model.add(LSTM(units=64,
                return_sequences=True,
                input_shape=(None, num_x_signals,)))
model.add(Dropout(dropout))
#model.add(TimeDistributed(Dense(units=5)))



# model.add(Dense(num_y_signals, activation='sigmoid'))

#linear statt sigmoid


model.add(Dense(num_y_signals,
                activation='linear',
                kernel_initializer=init))

warmup_steps = 50

#optimizer
optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)

print(model.summary())


#tensorboard setup
path_checkpoint = 'checkpoint.keras'
#validation loss abspeichern
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./logs/',
                                   histogram_freq=0,
                                   write_graph=True)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

#das Modell, für trainingssession
if fit == True:
    model.fit(x=generator,
                        epochs=10,
                        steps_per_epoch=30,
                        validation_data=validation_data,
                        callbacks=callbacks)
#falls checkpoint exisitert wird dieser zu validierung verwendet
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
#ergebniss validieren


result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))
result_2.append(result)
print("loss (test-set):", result)

print(result_2)print(result_2)