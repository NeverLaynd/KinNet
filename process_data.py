from torch.utils.data import Dataset
import torch
from rdkit.Chem import MACCSkeys,rdMolDescriptors
from rdkit import Chem
from joblib import Parallel, delayed
from rdkit.Chem import AllChem

from compound_process import smiles_to_graph

from torch_geometric.data import InMemoryDataset

from concurrent.futures import ThreadPoolExecutor
import os

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
            "U": 19, "T": 20, "W": 21, 
            "V": 22, "Y": 23, "X": 24, 
            "Z": 25 }

CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

MACCSLEN = 2

def label_chars(chars, max_len=1000, char_set=CHARPROTSET):
    X = torch.zeros(max_len, dtype=torch.long)
    for i, ch in enumerate(chars[:max_len]):
        X[i] = char_set[ch]
    return X


def slabel_chars(chars, max_len=1000, char_set=CHARISOSMISET):
    X = torch.zeros(max_len, dtype=torch.long)
    for i, ch in enumerate(chars[:max_len]):
        X[i] = char_set[ch]
    return X

def smiles_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
# ECFP4指纹
    # fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024) 
# pubchem指纹
    # fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=881) 
#  MACCS指纹
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor([int(_) for _ in fp.ToBitString()[1:]])

# 定义一个函数来处理smiles_fingerprint
def process_smiles_fingerprint(s):
    return smiles_fingerprint(s)

# 定义一个函数来处理label_chars
def process_label_chars(f):
    return label_chars(f, 1000, CHARPROTSET)

def smile_pt_file(fir,name):
    data =  torch.load(f"{fir}/{name}.pt")
    data.x = data.x.to(torch.float32)
    data.edge_attr = data.edge_attr.to(torch.float32)
    return data

def prot_pt_file(fir,name):
    name = name.split(".")[0]
    data =  torch.load(f"{fir}/{name}.pt")
    data.x = data.x.to(torch.float32)
    data.edge_attr = data.edge_attr.to(torch.float32)
    return data


class GNNDataset(InMemoryDataset):
    def __init__(self, root, index=0, types='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[index])
            x=0
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[index + len(self.raw_paths)])    

    @property
    def raw_file_names(self):
        return ['0/train.csv', '1/train.csv', '2/train.csv', '3/train.csv', '4/train.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_fold_0.pt', 'processed_data_fold_1.pt', 'processed_data_fold_2.pt', 'processed_data_fold_3.pt','processed_data_fold_4.pt', 
        'processed_test_data_fold_0.pt', 'processed_test_data_fold_1.pt', 'processed_test_data_fold_2.pt', 'processed_test_data_fold_3.pt','processed_test_data_fold_4.pt']

    def download(self):
        pass


class DTAData(Dataset):

    def __init__(self, data, device, seed=None,max_smiles_len=100, max_fasta_len=1000):
        self.smiles = data["SMILES"].values.tolist()
        self.fasta = data["Sequence"].values.tolist()

        #Acession Number,Gene,Kinase,Sequence,Compound,PubChem_Cid,SMILES,Kd
        self.smile_id = data["Kinase"].values.tolist()
        self.fasta_id = data["PubChem_Cid"].values.tolist()

        self.device = device
        self.max_smiles_len = max_smiles_len
        self.max_fasta_len = max_fasta_len

        self.smiles_fp = Parallel(n_jobs=16,verbose=1)(delayed(process_smiles_fingerprint)(s) for s in self.smiles)
        self.prot_cha = Parallel(n_jobs=16,verbose=1)(delayed(process_label_chars)(s) for s in self.fasta)

        
        self.cg =  Parallel(n_jobs=16,verbose=1)(delayed(smiles_to_graph)(s) for s in self.smiles)

        self.label = torch.tensor(data["Kd"].values.tolist(), dtype=torch.float32)


        print(len(self.smiles_fp))



    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.smiles_fp[idx],
                self.prot_cha[idx],
                self.label[idx],
                self.cg[idx],
                self.smile_id[idx],
                self.fasta_id[idx],
               )

