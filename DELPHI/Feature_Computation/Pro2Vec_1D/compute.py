import operator
import os
import numpy as np
import time
import sys
import argparse
import math

Dict_3mer_to_100vec = {}
# dim: delimiter
def get_3mer_and_np100vec_from_a_line(line, dim):
    np100 = []
    line = line.rstrip("\n").rstrip(" ").split(dim)
    # print (line)
    three_mer = line.pop(0)
    # print(three_mer)
    # print (line)
    np100 += [float(i) for i in line]
    np100 = np.asarray(np100)
    # print(np100)
    return three_mer, np100


def LoadPro2Vec():
    # TODO: we're going to need to fix this...
    import pathlib
    filename = (
        str(pathlib.Path().absolute())
        + "/DELPHI/Feature_Computation/Pro2Vec_1D/protVec_100d_3grams.csv"
    )
    f = open(filename, "r")

    while True:
        line = f.readline()
        if not line:
            break
        three_mer, np100vec = get_3mer_and_np100vec_from_a_line(line, "\t")
        Dict_3mer_to_100vec[three_mer] = np100vec


def GetFeature(ThreeMer, Feature_dict):
    if ThreeMer not in Feature_dict:
        print("[warning]: Feature_dict can't find ", ThreeMer, ". Returning 0")
        return 0
    else:
        return Feature_dict[ThreeMer]


def RetriveFeatureFromASequence(seq, Feature_dict):
    seq = seq.rstrip("\n").rstrip(" ")
    assert len(seq) >= 3
    Feature = []
    for index, item in enumerate(seq):
        sta = index - 1
        end = index + 1
        if (sta < 0) or (end >= len(seq)):
            Feature.append(Feature_dict["<unk>"])
        else:
            Feature.append(GetFeature(seq[sta : sta + 3], Feature_dict))
    return Feature


def load_fasta_and_compute(seq_fn, out_fn, Feature_dict):
    fin = open(seq_fn, "r")
    fout = open(out_fn, "w")
    while True:
        line_Pid = fin.readline()
        line_Pseq = fin.readline()
        if not line_Pseq:
            break
        fout.write(line_Pid)
        fout.write(line_Pseq)
        Feature = RetriveFeatureFromASequence(line_Pseq, Feature_dict)
        fout.write(",".join(map(str, Feature)) + "\n")
    fin.close()
    fout.close()


def main():
    # print("start")
    LoadPro2Vec()

    for key, value in Dict_3mer_to_100vec.items():
        Dict_3mer_to_100vec[key] = np.sum(value)
    # print(Dict_3mer_to_100vec["AAA"])
    max_key = max(Dict_3mer_to_100vec.keys(), key=(lambda k: Dict_3mer_to_100vec[k]))
    min_key = min(Dict_3mer_to_100vec.keys(), key=(lambda k: Dict_3mer_to_100vec[k]))
    max_value = Dict_3mer_to_100vec[max_key]
    min_value = Dict_3mer_to_100vec[min_key]
    # print(max_value)
    # print(min_value)
    for key, value in Dict_3mer_to_100vec.items():
        Dict_3mer_to_100vec[key] = (Dict_3mer_to_100vec[key] - min_value) / (
            max_value - min_value
        )
    # print(Dict_3mer_to_100vec["AAA"])
    # for key in Dict_3mer_to_100vec:
    #     print (key,": ", Dict_3mer_to_100vec[key])
    seq_fn = sys.argv[1]
    out_fn = sys.argv[2]
    # change the function below so that it loads 3mer
    load_fasta_and_compute(seq_fn, out_fn, Dict_3mer_to_100vec)
    # print("end")


if __name__ == "__main__":
    main()
