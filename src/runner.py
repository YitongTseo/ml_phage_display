import pathlib

HOME_DIRECTORY = pathlib.Path().absolute()
import pdb
from preprocessing.data_loading import read_data, preprocessing
from models.neural_nets import train


R3_lib = read_data(datafile="12ca5-MDM2-mCDH2-R3.csv")
X, y = preprocessing(R3_lib)
result = train(X, y)
pdb.set_trace()
print('now what...')
