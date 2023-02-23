import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def filter_to_nmers(seqs, nmer_len=9):
    ninemer_seq = []
    for seq in seqs:
        if len(seq) == nmer_len:
            ninemer_seq.append(seq[1:])
    return ninemer_seq

def generate_heatmap(seqs, title=None):
    ninemer_seq = filter_to_nmers(seqs, nmer_len=9)

    seq9_s = pd.Series(ninemer_seq)
    seq9_s = seq9_s.str.split(r"", expand=True)
    seq9_s = seq9_s.drop(0, axis=1)
    seq9_s = seq9_s.drop(9, axis=1)

    frames = []
    for i in range(8):
        frames.append(seq9_s.iloc[:, i].value_counts().sort_index())
    seq9_result = pd.concat(frames, axis=1, join="outer")
    seq9_result

    fig, ax = plt.subplots(figsize=(5, 10))
    hm = sns.heatmap(seq9_result, cmap=sns.color_palette("flare", as_cmap=True))
    if title:
        fig.suptitle(title)
    return hm
