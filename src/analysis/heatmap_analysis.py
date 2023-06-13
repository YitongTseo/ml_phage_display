import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.preprocessing.X_representation_utils import AMINO_ACIDS


def filter_to_nmers(seqs, nmer_len=9):
    ninemer_seq = []
    for seq in seqs:
        if len(seq) == nmer_len:
            ninemer_seq.append(seq[1:])
    return ninemer_seq


def generate_heatmap(seqs, title=None, vmax=300, vmin=0, drop_C=True, colormap="Greys"):
    ninemer_seq = filter_to_nmers(seqs, nmer_len=9)

    seq9_s = pd.Series(ninemer_seq)
    seq9_s = seq9_s.str.split(r"", expand=True)
    seq9_s = seq9_s.drop(0, axis=1)
    seq9_s = seq9_s.drop(9, axis=1)

    frames = []
    all_aa = pd.Series(index=list('ARNDCQEGHILKMFPSTWYV'))
    for i in range(8):
        value_counts = seq9_s.iloc[:, i].value_counts()
        # Add rows for missing amino acids
        value_counts=  all_aa.combine(
            value_counts,
            lambda s1, s2: s1 if s1 > s2 else s2,
            fill_value=None,
        )
        # Sort index and add to list of frames
        frames.append(value_counts.sort_index())

    seq9_result = pd.concat(frames, axis=1, join="outer")
    if drop_C:
        print("drop C")
        seq9_result = seq9_result.drop("C")
    print("max is ", seq9_result.max().max())

    fig, ax = plt.subplots(figsize=(5, 10))
    hm = sns.heatmap(
        seq9_result,
        cmap=sns.color_palette(colormap, as_cmap=True),
        vmax=vmax,
        vmin=vmin,
    )
    if title:
        fig.suptitle(title)

    # Get the colorbar object from the heatmap
    cbar = hm.collections[0].colorbar

    # Set the ticks to only show the maximum and minimum values
    cbar.set_ticks([vmin, vmax])
    return hm
