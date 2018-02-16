import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow as tf
def create_lstm_vae(input_dim, 
     timesteps,
    batch_size, 
    intermediate_dim, 
    latent_dim,
    epsilon_std=1.):

    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM. 
        latent_dim: int, latent z-layer shape. 
        epsilon_std: float, z-layer sigma.
    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(intermediate_dim,return_sequences=True,activation='tanh')(x)
    h = LSTM(int(intermediate_dim/2),activation='tanh')(h)
    # VAE Z layer
    z_mean = Dense(latent_dim,activation='linear')(h)
    z_log_sigma = Dense(latent_dim,activation = 'linear')(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling)([z_mean, z_log_sigma])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True,activation='tanh')
    decoder_h1 =  LSTM(int(intermediate_dim/2), return_sequences=True,activation='tanh')
    decoder_mean = Dense(input_dim,activation='softmax')

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)
    h_decoded1 = decoder_h1(h_decoded)
    
    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded1)
    
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)
    _h_decoded = decoder_h1(_h_decoded)
    
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss,metrics=['mse'])
    
    return vae, encoder, generator

read = True
if read:
    Y = pd.read_pickle('save_labels_from_Jan17_85.p')
    X = pd.read_pickle('save_data_from_Jan17_85.p')



x_model = pd.DataFrame(data=X)

drop_col = False

if drop_col:
    count_real_values = x_model.count()
    count_real_values = ((count_real_values) * 100) / len(x_model)
    sel_col = count_real_values[count_real_values > 30]
    sel_col = sel_col.index

    x_model = x_model[sel_col]

max_x_model = x_model.max()
min_x_model = x_model.min()

fmax = 1
fmin = -1
x_model1 = (x_model - min_x_model) / (max_x_model - min_x_model)
x_model1 = x_model1 * (fmax - fmin) + fmin

#x_model1 = x_model1.fillna(0)
x_model1 = x_model1.interpolate(limit_direction='both')
from sklearn.model_selection import train_test_split

X = np.array(x_model1)
Y = np.array(Y)

import sklearn

X_lstm = np.zeros((len(X), 25, 6))

for i in range(0, len(X)):
    a = X[i].reshape((6, 25)).T
    X_lstm[i] = a

X_train, X_test, Y_train, Y_test = train_test_split(X_lstm, Y, test_size=0.1, random_state=None)


X_train = X_train[:37952]
X_test = X_test[:4192]

input_dim = 6
timesteps = 25 
batch_size = 32
vae, enc, gen = create_lstm_vae(input_dim,
    timesteps = 25,
    batch_size=batch_size, 
    intermediate_dim=16,
    latent_dim=8,
    epsilon_std=1.)


csv_logger = CSVLogger('log.csv', append=True, separator=';')
checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

vae.fit(X_train, X_train,epochs=1000, validation_data=(X_test, X_test),verbose=2,callbacks=[csv_logger, checkpointer, early_stopping,TensorBoard(log_dir='/tmp/autoencoder')])

vae.save('vae_LSTM_16_10_adam.h5')