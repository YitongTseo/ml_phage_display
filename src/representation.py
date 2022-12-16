# TODO: we probably want to make this HOME_DIRECTORY specification a bit cleaner
import pathlib

import sys
import os

sys.path.append(os.path.join(pathlib.Path().absolute(), "DELPHI/Feature_Computation"))
import numpy as np
import Pro2Vec_1D.compute as Pro2Vec
from utils.representation_utils import (
    initialize_Proc2Vec_embeddings,
    initialize_RAAs,
    initialize_physiochemical_properties,
    initialize_mol_descriptor_calculator,
    initialize_aa_to_smiles,
)
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

MAX_PEPTIDE_LEN = 14
# Initialize all per residue representations just once globally
# for efficient lookup
DICT_3MER_TO_100VEC = initialize_Proc2Vec_embeddings()
DICT_AA_TO_RAA, DICT_RAA_TO_AA = initialize_RAAs()
EXTENDED_PROP_AA = initialize_physiochemical_properties()
AA_TO_SMILES = initialize_aa_to_smiles()
MOL_DESCRIPTOR_CALCULTOR = initialize_mol_descriptor_calculator()


def seq_to_pro2vec(seq):
    vec = np.zeros(MAX_PEPTIDE_LEN)
    features = Pro2Vec.RetriveFeatureFromASequence(seq, DICT_3MER_TO_100VEC)
    vec[: len(seq)] = np.array(features)
    return vec


def seq_to_RAA(seq):
    vec = np.zeros(MAX_PEPTIDE_LEN)
    features = []
    for aa in seq:
        features.append(DICT_AA_TO_RAA[aa])
    vec[: len(seq)] = np.array(features)
    return vec


def seq_to_prop(seq):
    vec = np.zeros((MAX_PEPTIDE_LEN, MAX_PEPTIDE_LEN))
    for idx, aa in enumerate(seq):
        vec[idx, :] = EXTENDED_PROP_AA[aa]
    return vec


def RAA_to_seq(RAA_ls):
    seq = ""
    for RAA in RAA_ls:
        seq += DICT_RAA_TO_AA[RAA]
    return seq


"""
These two representations (seq_to_fp, seq_to_desc) aren't used for the time being
"""


def seq_to_fp(seq):
    fp_list = np.zeros((MAX_PEPTIDE_LEN, 512))
    for idx, aa in enumerate(seq):
        m = Chem.MolFromSmiles(AA_TO_SMILES[aa])
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=512)
        arr = np.zeros((0,), dtype=np.int8)
        a = DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list[idx] = arr
    return fp_list


def seq_to_desc(seq):
    mol = Chem.MolFromSequence(seq)
    vec = np.array(list(MOL_DESCRIPTOR_CALCULTOR.CalcDescriptors(mol)))
    return vec
