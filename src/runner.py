import pdb
from preprocessing.data_loading import read_data, preprocessing
from models.experiment import run_adhoc_experiment
from models.rnn import RNN


R3_lib = read_data(datafile="12ca5-MDM2-mCDH2-R3.csv")
X, y = preprocessing(R3_lib, protein_of_interest='MDM2')
result = run_adhoc_experiment(X, y, RNN)
# result = run_cross_validation_experiment(X, y, RNN)
pdb.set_trace()
print('now what...')
