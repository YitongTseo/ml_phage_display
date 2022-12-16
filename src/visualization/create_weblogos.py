import pandas as pd
import pdb
import matplotlib.pyplot as plt
import pdb
import subprocess
import logomaker as lm
import matplotlib.pyplot as plt
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
from enum import Enum


class Y_AXIS_UNIT(Enum):
    COUNTS = 1
    BITS = 2


def muscle_align(seqs, seq_record_name, align_name):
    SeqIO.write(
        [SeqRecord(Seq(seq), id=seq) for seq in seqs],
        seq_record_name,
        "fasta",
    )
    subprocess.call(
        "muscle -super5 %s -output %s" % (seq_record_name, align_name),
        shell=True,
    )
    with open(align_name, "r") as f:
        raw_seqs = f.readlines()
    return [seq.strip() for seq in raw_seqs if ("#" not in seq) and (">") not in seq]


def save_web_logo_alignment(
    seqs,
    axis,
    seq_record_name="_example.fasta",
    align_name="_align.fasta",
    web_logo_name="logo.png",
    to_type="counts",
    align=True,
):
    if align:
        seqs = muscle_align(seqs, seq_record_name, align_name)
    counts_mat = lm.alignment_to_matrix(seqs, to_type=to_type)
    counts_mat.head()
    logo = lm.Logo(counts_mat, ax=axis, color_scheme="hydrophobicity")
    plt.savefig(web_logo_name)


def generate_weblogos_by_library(
    df, target_protein_name, y_axis_units=Y_AXIS_UNIT.BITS, library_count_threshold=1
):
    def peptide_passes(seq):
        cysteine_positions = [idx for idx, i in enumerate(seq) if i == "C"]
        # We only want completely linear sequences (no Cysteines)
        # Or well behaving macrocylcles (only 2 Cysteines)
        return len(cysteine_positions) == 0 or len(cysteine_positions) == 2

    def get_library_name(seq):
        cysteine_positions = [idx for idx, i in enumerate(seq) if i == "C"]
        if len(cysteine_positions) >= 2:
            return (
                "Macrocyclic "
                + str(len(seq))
                + "mers"
                + " w/ C's @ "
                + str(cysteine_positions)
            )
        else:
            # For Peptides with only 1 or 0 Cysteines, we should just treat them as linear
            return "Linear " + str(len(seq)) + "mers"

    df = df[df["Sequence"].apply(lambda seq: peptide_passes(seq))]
    # Create a key that takes into account len & cysteine placements within peptide
    macrocycle_idx_df = pd.DataFrame(
        df["Sequence"].apply(lambda seq: get_library_name(seq))
    )
    library_to_indices = macrocycle_idx_df.groupby("Sequence").indices
    # Collect Dict into a List & Sort by relevance, discard single library matches
    # (aka first element in library_and_seqs_list represents the most common library)
    library_and_seqs_list = sorted(
        [
            (lib, df["Sequence"].iloc[idxs])
            for lib, idxs in library_to_indices.items()
            if len(idxs) > library_count_threshold
        ],
        key=lambda tup: len(tup[1]),
        reverse=True,
    )
    fig, axs = plt.subplots(len(library_and_seqs_list), 1)

    for idx, (lib_name, seqs) in enumerate(library_and_seqs_list):
        axis = axs[idx]
        axis.set_title(
            lib_name + " (" + str(len(seqs)) + " peptides)",
            size=10,
        )
        axis.set_xticks([])
        if y_axis_units == Y_AXIS_UNIT.BITS:
            save_web_logo_alignment(
                seqs=seqs, align=True, axis=axis, to_type="information"
            )
        elif y_axis_units == Y_AXIS_UNIT.COUNTS:
            save_web_logo_alignment(seqs=seqs, align=True, axis=axis, to_type="counts")

    fig.suptitle(target_protein_name + " Weblogos by Library", fontsize=15)
    fig.subplots_adjust(hspace=0.5)
    fig.add_subplot(111, frameon=False)
    # hide y & x tick labels of the big axis
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("Alignment Position")
    if y_axis_units == Y_AXIS_UNIT.BITS:
        plt.ylabel("Information (Bits)")
    elif y_axis_units == Y_AXIS_UNIT.COUNTS:
        plt.ylabel("Residue Positional Counts")
    plt.show()


generate_weblogos_by_library(
    pd.read_csv("mdm2_good.txt"),
    target_protein_name="MDM2",
    y_axis_units=Y_AXIS_UNIT.BITS,
)
generate_weblogos_by_library(
    pd.read_csv("12ca5_good.txt"),
    target_protein_name="12ca5",
    library_count_threshold=3,
    y_axis_units=Y_AXIS_UNIT.BITS,
)

generate_weblogos_by_library(
    pd.read_csv("mdm2_good.txt"),
    target_protein_name="MDM2",
    y_axis_units=Y_AXIS_UNIT.COUNTS,
)
generate_weblogos_by_library(
    pd.read_csv("12ca5_good.txt"),
    target_protein_name="12ca5",
    library_count_threshold=3,
    y_axis_units=Y_AXIS_UNIT.COUNTS,
)
