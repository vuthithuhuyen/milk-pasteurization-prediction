from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

from Helper.MyPathFunctions import GetCWD
import pathlib


appPath = GetCWD().parent
mainWindowUI = appPath / 'View' / 'main.ui'


raw_data_file = appPath / 'Data' / 'pasteurizer.csv'
inputfeatures = ['MIXA_PASTEUR_STATE', 'MIXB_PASTEUR_STATE', 'MIXA_PASTEUR_TEMP', 'MIXB_PASTEUR_TEMP']
output = ['INSP']
cols = inputfeatures + output


scaler = MinMaxScaler()
n_inputs, n_outputs = 1, 1
n_features = len(inputfeatures)
name_to_int, int_to_name = None, None
load_trained = False

mainwindow = appPath / 'View' / 'mainwindow.ui'
analysis_data = None

X, y = None, None
display_rescale = 60
row_to_predict = None



