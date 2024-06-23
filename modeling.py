import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
import joblib
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
train = pd.read_pickle('train.pkl')
val = pd.read_pickle('val.pkl')
test = pd.read_pickle('test.pkl')


x_columns = train.columns.values.tolist()
for col in ['date', 'name','30dGain', 'code','Y','priceTo每股公积金']:
    x_columns.remove(col)
y_columns = ['Y']

x_train = train[x_columns].values
y_train = np.array(train[y_columns].values.ravel().tolist())
x_val = val[x_columns].values
y_val = np.array(val[y_columns].values.ravel().tolist())

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)

x_val = scaler.transform(x_val)

joblib.dump(scaler, 'scaler.joblib')

# 原始模型
model = Sequential([
    # 第一个卷积层
    Dense(512,  activation='tanh', input_shape=(x_train.shape[1],)),
    Dropout(0.2),
    Dense(1024, activation='tanh', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1024, activation='tanh', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(512, activation='tanh', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(64, activation='tanh'),
    Dense(8, activation='tanh'),
    Dense(3, activation='softmax') # 最初的输出层
])
optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max', restore_best_weights=True)

feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

model.fit(x_train, y_train, epochs=10000, batch_size=128, validation_data=(x_val, y_val), callbacks=[early_stopping])

feature_model.save('feature_extraction.h5')
