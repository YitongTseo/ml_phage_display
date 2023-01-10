# TODO: we probably want to make this HOME_DIRECTORY specification a bit cleaner
import pathlib

HOME_DIRECTORY = pathlib.Path().absolute().parent

import sys
import os
import pdb
import pandas as pd

sys.path.append(os.path.join(HOME_DIRECTORY, "DELPHI/Feature_Computation"))
import numpy as np
import Pro2Vec_1D.compute as Pro2Vec
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

"""
    Taken from table 5 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8232778/ 
    H11 and H12: hydrophobicity; H2: hydrophilicity; NCI: net charge index of side chains; 
    P11 and P12: polarity; P2: polarizability; SASA: solvent-accessible surface area; 
    V: volume of side chains; F: flexibility; A1: accessibility; E: exposed; T: turns; A2: antigenic. 
    * Hydrophobicity (H11 and H12) and polarity (P11 and P12) were calculated using two methods.
"""
AA_PROPERTY_FILE = "src/preprocessing/amino_acid_properties.tsv"
AA_PROPERTY_ORDERING = [
    "H11",
    "H12",
    "H2",
    "NCI",
    "P11",
    "P12",
    "P2",
    "SASA",
    "V",
    "F",
    "A1",
    "E",
    "T",
    "A2",
]
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVX"


def initialize_physiochemical_properties():
    extended_prop_aa_df = pd.read_csv(AA_PROPERTY_FILE, delimiter="\t")
    extended_prop_aa_df = extended_prop_aa_df.set_index("AA")
    assert all(extended_prop_aa_df.columns == AA_PROPERTY_ORDERING)

    extended_prop_aa_dict = extended_prop_aa_df.T.to_dict(orient="list")
    for key in extended_prop_aa_dict.keys():
        extended_prop_aa_dict[key] = np.array(extended_prop_aa_dict[key])
    return extended_prop_aa_dict


def initialize_RAAs():
    """
    Relative Amino Acid Propensity for Binding scores taken from
    Chen KH, Hu YJ. Residue-Residue Interaction Prediction via Stacked Meta-Learning. Int J Mol Sci. 2021
    """
    # fmt: off
    Dict_aa_to_RAA = [0.19230769, 0.32051282, 0.1474359, 0.03205128, 0.73076923, 0.17307692, 0.02564103, 0.08333333, 0.35897436, 0.69871795, 0.63461538, 0.00000001, 0.83333333, 1., 0.13461538, 0.16025641, 0.19871795, 0.8525641, 0.69871795, 0.48076923, 0.3878]
    # fmt: on
    Dict_aa_to_RAA = {AMINO_ACIDS[i]: Dict_aa_to_RAA[i] for i in range(21)}

    # TODO(Yitong): why do we need this reverse map?
    Dict_RAA_to_aa = {y: x for x, y in Dict_aa_to_RAA.items()}
    Dict_RAA_to_aa[0.0] = ""
    return Dict_aa_to_RAA, Dict_RAA_to_aa


def initialize_Proc2Vec_embeddings():
    Pro2Vec.LoadPro2Vec()
    Dict_3mer_to_100vec = Pro2Vec.Dict_3mer_to_100vec
    for key, value in Dict_3mer_to_100vec.items():
        Dict_3mer_to_100vec[key] = np.sum(value)

    max_key = max(Dict_3mer_to_100vec.keys(), key=(lambda k: Dict_3mer_to_100vec[k]))
    min_key = min(Dict_3mer_to_100vec.keys(), key=(lambda k: Dict_3mer_to_100vec[k]))
    max_value = Dict_3mer_to_100vec[max_key]
    min_value = Dict_3mer_to_100vec[min_key]
    for key, value in Dict_3mer_to_100vec.items():
        Dict_3mer_to_100vec[key] = (Dict_3mer_to_100vec[key] - min_value) / (
            max_value - min_value
        )
    return Dict_3mer_to_100vec


def initialize_aa_to_smiles():
    return {
        "A": "C[C@H](N)C=O",
        "C": "N[C@H](C=O)CS",
        "D": "N[C@H](C=O)CC(=O)O",
        "E": "N[C@H](C=O)CCC(=O)O",
        "F": "N[C@H](C=O)Cc1ccccc1",
        "G": "NCC=O",
        "H": "N[C@H](C=O)Cc1c[nH]cn1",
        "I": "CC[C@H](C)[C@H](N)C=O",
        "K": "NCCCC[C@H](N)C=O",
        "L": "CC(C)C[C@H](N)C=O",
        "M": "CSCC[C@H](N)C=O",
        "N": "NC(=O)C[C@H](N)C=O",
        "P": "O=C[C@@H]1CCCN1",
        "Q": "NC(=O)CC[C@H](N)C=O",
        "R": "N=C(N)NCCC[C@H](N)C=O",
        "S": "N[C@H](C=O)CO",
        "T": "C[C@@H](O)[C@H](N)C=O",
        "V": "CC(C)[C@H](N)C=O",
        "W": "N[C@H](C=O)Cc1c[nH]c2ccccc12",
        "Y": "N[C@H](C=O)Cc1ccc(O)cc1",
    }


def initialize_mol_descriptor_calculator():
    # fmt: off
    chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 
                        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 
                        'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 
                        'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 
                        'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 
                        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 
                        'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
                        'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 
                        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 
                        'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
                        'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 
                        'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 
                        'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9'
                        ]
    # fmt: on
    return MolecularDescriptorCalculator(chosen_descriptors)
