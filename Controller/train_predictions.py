from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Model.GlobalVariables import n_features
from Helper.DataPreprocessing import ReadData, ScaleData

# Read raw .csv data to X-Input and y-output
from Model.DeeplearningModel import prepare_mlp_model, EvaluateResult
import pandas as pd

df, X, y = ReadData()

#----------Temp--------------------------------
# plt.figure()
# # plt.plot(X[X.columns[2]], label="MixA_temprature")
# plt.plot(X[X.columns[2]], label=inputfeatures[2])
# plt.xlabel("Samples")
# plt.ylabel("Temprature")
# plt.grid()
# plt.legend()
# plt.show()
#---------------------------------------------


# Scale data
X, y, name_to_int, int_to_name = ScaleData(X, y)

#----------Temp--------------------------------
# plt.figure()
# plt.plot(X[:, 2], label=inputfeatures[2])
# plt.plot(X[:, 3], label=inputfeatures[3])
# plt.xlabel("Samples")
# plt.ylabel("Scaled temprature")
# plt.grid()
# plt.legend()
# plt.show()
#---------------------------------------------


# split train, test
split = train_test_split(X, y, test_size=0.7)
(train_X, test_X, train_y, test_y) = split


# create model
model = prepare_mlp_model(n_features, name_to_int)
print(model.summary())
#
# train_X = train_X.reshape(-1, 1, train_X.shape[1])
# train_y = train_y.reshape(-1, 1, train_y.shape[1])
print(train_X.shape, train_y.shape)
plt.figure()

history = model.fit(train_X, train_y, epochs=200, verbose=2, batch_size=30)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.legend()


predict = model.predict(test_X)
predict_result = EvaluateResult(predict, test_y, name_to_int)
print(f' predict result {predict_result}')
plt.show()


