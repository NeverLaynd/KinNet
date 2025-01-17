import torch
import pandas as pd 
import time

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from logger import create_logger

from models.model import Classifier, FASTAModel2, FASTAModel_Smiles
from models.model_graphkan import Drug3DModel
from process_data import DTAData, CHARPROTLEN, MACCSLEN
from train import train
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
code_path = os.path.dirname(os.path.realpath(__file__))
print('code_path:', code_path)
result_path = code_path + '/result/'
if not os.path.exists(result_path):
    os.makedirs(result_path)


MODEL_NAME = "GKMDTA"
BATCH_SIZE = 128
DATASET = "davis"                                  

writer = SummaryWriter(log_dir="/data/GKMDTA")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if DATASET == 'davis':
        data_path = f"/data/GKMDTA/data/davis"
        train_data = pd.read_csv(f"{data_path}/train.csv")
        valid_data = pd.read_csv(f"{data_path}/valid.csv")
        test_data = valid_data
        max_smiles_len = 85
        max_fasta_len = 1000

    output_dir = result_path
    date_now=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_dir = os.path.join(output_dir,date_now)+f"{DATASET}_{BATCH_SIZE}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    global logger

    date_now=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger,out_dir = create_logger(output_dir=output_dir,date_now = date_now, name=f"{DATASET}_{BATCH_SIZE}")
    logger.info(f"{date_now}_{DATASET}_mam_qs_mam3")

    train_set = DTAData(train_data,device, max_smiles_len, max_fasta_len)
    valid_set = DTAData(valid_data,device, max_smiles_len, max_fasta_len)
    test_set = DTAData(test_data,device, max_smiles_len, max_fasta_len)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=8)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    smiles_model = FASTAModel_Smiles(char_set_len = MACCSLEN)
    fasta_model = FASTAModel2(char_set_len=CHARPROTLEN+1)

    mol_model = Drug3DModel(d_vocab=44, d_edge=10, hidden_dim=166,
                            in_feat_dropout=0.0,  pos_enc_dim=8)
    model = Classifier(smiles_model, fasta_model,mol_model,device)
    model = model.to(device)

    logger.info(model)
    models_path = f"{result_path}/Output1_{DATASET}_{BATCH_SIZE}"
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    train(model, train_loader, valid_loader, test_loader, writer, MODEL_NAME,models_path,device,logger)

    del model

if __name__ == "__main__":
    main()