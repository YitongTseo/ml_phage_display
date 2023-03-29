import tensorflow as tf
import os
import random
import numpy as np
from keras import backend as K

def find_3mer(seq, tmer):
    for i in range(len(seq) - 2):
        if seq[i] == tmer[0] and seq[i + 1] == tmer[1] and seq[i + 2] == tmer[2]:
            return True
    return False

def cnt_c(seq):
    cnt = 0
    for i in range(len(seq)):
        if seq[i] == "C":
            cnt += 1
    return cnt


def seed_everything(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
