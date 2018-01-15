from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping
from sklearn.grid_search import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout

params = {
    
    'encoding_dim':[120,128,136,144],
     'epochs' : [1000],
    'batch_size' :[32,128,256]
}

def create_model(encoding_dim = 20):
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim, ))
    encoded = Dense(encoding_dim,activity_regularizer=regularizers.l1(10e-5),activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(int(encoding_dim/2), activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(int(encoding_dim/4), activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    decoded = Dense(int(encoding_dim/4), activation='relu')(encoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(int(encoding_dim/2), activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error',metrics=['accuracy'])
    return autoencoder

model = KerasClassifier(build_fn=create_model)


random = RandomizedSearchCV(estimator=model, param_distributions=params)
random_result = random.fit(X_train,X_train)
# summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
