import pathlib
import sys
HOME_DIRECTORY = pathlib.Path().absolute()
sys.path.append(str(HOME_DIRECTORY) + '/src')

import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import umap


def embedding_classification(model, X_train):
  nn_emb = model.layers[2](model.layers[1](model.layers[0](X_train)))
  reducer = umap.UMAP(n_neighbors=10,
                    min_dist=0.1,
                    n_components=2)
  reduced_emb = reducer.fit_transform(nn_emb)
  return reduced_emb
  
def embedding_regression(model, X_train):
  nn_emb =  model.layers[11](model.layers[10](model.layers[9](model.layers[8](model.layers[7](model.layers[6](model.layers[5](model.layers[4](model.layers[3](model.layers[2](model.layers[1](model.layers[0](X_train))))))))))))
  reducer = umap.UMAP(n_neighbors=10,
                    min_dist=0.1,
                    n_components=2)
  reduced_emb = reducer.fit_transform(nn_emb)
  return reduced_emb

# tests label with FC value
def UMAP_log_Fold(embedding, y_train):
  ax = sns.scatterplot(
    x=embedding[:, 0],
    y=embedding[:, 1],
    hue=y_train[:,1], alpha=0.1)
  plt.title("RNN embedding UMAP, label with log FC value")


# label with binary log FC value
def UMAP_binary_log_Fold(embedding, y_train):
  ax = sns.scatterplot(
      x=embedding[:, 0],
      y=embedding[:, 1],
      hue=y_train>0, alpha=0.1)
  plt.title("RNN embedding UMAP, label with binary log FC value")
  

# tests label with -log P value 
def UMAP_log_P(embedding, z_train):
  ax = sns.scatterplot(
      x=embedding[:, 0],
      y=embedding[:, 1],
      hue=-z_train, alpha=0.1)
  plt.title("RNN embedding UMAP, label with - log P value")
  
  
# tests label with binary -log P value 
def UMAP_binary_log_P(embedding, z_train):
  ax = sns.scatterplot(
      x=embedding[:, 0],
      y=embedding[:, 1],
      hue=-z_train>-np.log10(0.05), alpha=0.1)
  plt.title("RNN embedding UMAP, label with binary - log P value")
  
  
# testslabel with both log FC and -log P value 
def UMAP_joint(embedding, y_train,z_train):
  ax = sns.scatterplot(
      x=embedding[:, 0],
      y=embedding[:, 1],
      hue=(y_train>0)*(-z_train>-np.log10(0.05)), alpha=0.1)
  plt.title("RNN embedding UMAP, label with both log FC and - log P value")
  
  
 

# nn_emb = model.layers[2](model.layers[1](model.layers[0](X_train)))
# reducer = umap.UMAP(n_neighbors=10,
#                     min_dist=0.1,
#                     n_components=2)

# # get the 4th fold because the 4th model showed the best precision
# # actually unnecessary, can just use whatever fold
# kf = KFold(n_splits=2)
# i = 0
# y_reg = np.array(list(y_reg))
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     y_reg_train, y_reg_test = y_reg[train_index], y_reg[test_index]
#     if i==0:
#         break
#     i += 1
# embedding = reducer.fit_transform(nn_emb)

# # tests
# ax = sns.scatterplot(
#     x=embedding[:, 0],
#     y=embedding[:, 1],
#     hue=y_train, alpha=0.1)
# plt.title("MDM2 RNN embedding UMAP")

# # tests
# X_train_prop1 = X_train[:,:,2].mean(-1)
# ax = sns.scatterplot(
#     x=embedding[:, 0],
#     y=embedding[:, 1],
#     hue=X_train_prop1, alpha=0.1)
# plt.title("MDM2 RNN embedding UMAP")

# # color by volume of side chain
# X_train_prop2 = X_train[:,:,10].sum(-1)#.sum(-1)
# ax = sns.scatterplot(
#     x=embedding[:, 0],
#     y=embedding[:, 1],
#     hue=X_train_prop2, alpha=0.1)
# plt.title("MDM2 RNN embedding UMAP")

# # color by SASA
# X_train_prop2 = X_train[:,:,9].sum(-1)#.sum(-1)
# ax = sns.scatterplot(
#     x=embedding[:, 0],
#     y=embedding[:, 1],
#     hue=X_train_prop2, alpha=0.1)
# plt.title("MDM2 RNN embedding UMAP")

# # color by polarizability
# X_train_prop2 = X_train[:,:,8].sum(-1)
# ax = sns.scatterplot(
#     x=embedding[:, 0],
#     y=embedding[:, 1],
#     hue=X_train_prop2, alpha=0.1)
# plt.title("MDM2 RNN embedding UMAP")
