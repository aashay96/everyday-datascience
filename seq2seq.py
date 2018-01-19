from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

X_train,X_test = train_test_split(X,test_size=0.1,random_state=None)


inp = Input((None,6))

out = LSTM(units = 100, return_sequences=True, activation='tanh')(inp)
out = LSTM(units = 80, return_sequences=True)(out)
out = LSTM(units = 76, return_sequences=True, activation='tanh')(out)
out = LSTM(units = 72, return_sequences=False, activation='tanh')(out)
encoder = Model(inp,out)   

out_dec = RepeatVector(72)(out) # I also tried to use Reshape instead, not really a difference

out1 = LSTM(20,return_sequences=True, activation='tanh')(out_dec)   
out1 = LSTM(15,return_sequences=True, activation='tanh')(out1)   
out1 = LSTM(10,return_sequences=True, activation='tanh')(out1)   
out1 = LSTM(6,return_sequences=True, activation='sigmoid')(out1) # I also tried softmax instead of sigmoid, not really a difference

decoder = Model(inp,out1)

autoencoder = Model(encoder.inputs, decoder(encoder.inputs))

autoencoder.compile(loss='mean_squared_error',
              optimizer='RMSprop',
              metrics=['accuracy'])

autoencoder.fit(X_train, X_train,
          batch_size=8,
          epochs=100,
          validation_data=(X_test,X_test))