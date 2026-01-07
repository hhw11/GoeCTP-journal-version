import argparse
import torch
from torch import nn
import numpy as np
from dataset import get_dataset
import pandas as pd
import pickle as pk
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from jarvis.core.atoms import Atoms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from e3nn.io import CartesianTensor
from pandarallel import pandarallel
from dataset import get_symmetry_dataset
pandarallel.initialize(progress_bar=False)

from graphs_for_matbench import atoms2graphs,atoms2graphs_ic, GraphDataset
from utils import get_id_train_val_test
from GoeCTP import GoeCTP_plus
#from crystal_mamba2 import GoeCTP_plus
from megnet import MEGNET
from mace_models import MACE
from ecomformer import EComformerEquivariant,iComformer


from etgnn import DimeNetPlusPlusWrap
import matplotlib.pyplot as plt
from e3nn import o3
import pdb
# torch config
torch.set_default_dtype(torch.float32)
import torch
import numpy as np
import random
import os
import time
import torch.multiprocessing
from sklearn.model_selection import train_test_split
from matbench.bench import MatbenchBenchmark

torch.multiprocessing.set_sharing_strategy('file_system')

# Set the random seed for Python, NumPy, and PyTorch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# Ensuring CUDA's determinism
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
    # Configure PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

adptor = JarvisAtomsAdaptor()

diagonal = [0, 4, 8]
off_diagonal = [1, 2, 3, 5, 6, 7]
converter = CartesianTensor("ij")
irreps_output = o3.Irreps('1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o')


'''
def structure_to_graphs(
    df: pd.DataFrame,
    use_corrected_structure: bool = False,
    reduce_cell: bool = False,
    cutoff: float = 4.0,
    max_neighbors: int = 16
):
    def atoms_to_graph(p_input):
        """Convert structure dict to DGLGraph."""
        structure = adptor.get_atoms(p_input["structure"])
        return atoms2graphs(
            structure,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            reduce=reduce_cell,
            #equivalent_atoms=p_input['equivalent_atoms'],
            use_canonize=True,
        )
    #graphs = df["p_input"].parallel_apply(atoms_to_graph).values
    graphs = df["p_input"].parallel_apply(atoms_to_graph).values
    # graphs = df["p_input"].apply(atoms_to_graph).values
    return graphs
'''


def structure_to_graphs(
    df: pd.DataFrame,
    use_corrected_structure: bool = False,
    reduce_cell: bool = False,
    cutoff: float = 4.0,
    max_neighbors: int = 16
):
    def atoms_to_graph(structure):
        """Convert structure dict to DGLGraph."""
        structure = adptor.get_atoms(structure)
        return atoms2graphs_ic(
            structure,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_canonize=True,
            use_lattice = True,
            use_angle = False,
        )
    #graphs = df["p_input"].parallel_apply(atoms_to_graph).values
    graphs = df["structure"].apply(atoms_to_graph).values
    return graphs





def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.element_size() * parameter.nelement()
    for parameter in model.buffers():
        total_params += parameter.element_size() * parameter.nelement()
    total_params = total_params / 1024 / 1024
    print(f"Total size: {total_params}")
    print("Total trainable parameter number", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total_params


class PolynomialLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, start_lr, end_lr, power=1, last_epoch=-1):
        self.max_iters = max_iters
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power
        self.last_iter = 0  # Custom attribute to keep track of last iteration count
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            (self.start_lr - self.end_lr) * 
            ((1 - self.last_iter / self.max_iters) ** self.power) + self.end_lr 
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.last_iter += 1  # Increment the last iteration count
        return super().step(epoch)

def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def get_pyg_dataset(data, target, reduce_cell=False):
    df_dataset = pd.DataFrame(data)
    g_dataset = structure_to_graphs(df_dataset, reduce_cell=reduce_cell)
    pyg_dataset = GraphDataset(df=df_dataset,graphs=g_dataset, target=target)
    return pyg_dataset





def dielectric_to_refractive_index(dielectric_tensor: torch.Tensor) -> torch.Tensor:

    diag_elements = torch.diagonal(dielectric_tensor, dim1=1, dim2=2)  # [batch_size, 3]
    

    eps_iso = diag_elements.mean(dim=1)  # [batch_size]
    #eps = 1e-8
    #eye = torch.eye(3, device=dielectric_tensor.device, dtype=dielectric_tensor.dtype)
    #dielectric_tensor = dielectric_tensor + eps * eye    

    #refractive_index = torch.sqrt(eps_iso)
    refractive_index=eps_iso
    
    #vals, _ = torch.linalg.eigh(dielectric_tensor)   
    
    #n_iso = vals.mean(dim=1)  
    #refractive_index = torch.sqrt(n_iso)        
    return refractive_index


def train(model, args):



        
    mb = MatbenchBenchmark(autoload=False,subset=["matbench_dielectric"])
    
    for task in mb.tasks:
        task.load()
        ii=0
        for fold in task.folds:
            ii=ii+1
            
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
  
            
            
            x_train, x_val, y_train, y_val, k_train, k_val = train_test_split(
                train_inputs.tolist(), train_outputs.tolist(), train_inputs.index.tolist(), test_size=0.1, random_state=123321)    
            dataset_train = []
            for i in tqdm(range(len(x_train))):           
                temp={'structure':x_train[i],'target':y_train[i]}
                dataset_train.append(temp)                  


            dataset_val = []
            for i in tqdm(range(len(x_val))):   
                temp={'structure':x_val[i],'target':y_val[i]}
                dataset_val.append(temp)            
                            

            pyg_dataset_train = get_pyg_dataset(dataset_train, args.subtarget, args.reduce_cell)
            pyg_dataset_val = get_pyg_dataset(dataset_val, args.subtarget, args.reduce_cell) 
            '''                
            with open(f"./matbenck_train_temp.pkl", 'wb') as f:
                pk.dump(pyg_dataset_train, f)
            with open(f"./matbenck_val_temp.pkl", 'wb') as f:
                pk.dump(pyg_dataset_val, f)     

            with open(f"./matbenck_train_temp.pkl", 'rb') as f:
                pyg_dataset_train = pk.load(f)
            with open(f"./matbenck_val_temp.pkl", 'rb') as f:
                pyg_dataset_val = pk.load(f)               
            '''                         
            collate_fn = pyg_dataset_train.collate
            train_loader = DataLoader(
                pyg_dataset_train,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True,
                num_workers=4,
                pin_memory=True,
            )
        
            val_loader = DataLoader(
                pyg_dataset_val,
                batch_size=64,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=True,
                num_workers=4,
                pin_memory=True,
            )
        
        
        
        
        
        
            model.to(device)
            
            #params = group_decay(model)    
            optimizer = torch.optim.AdamW(
                model.parameters(),#params
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            steps_per_epoch = len(train_loader)
            total_iter = steps_per_epoch * args.epochs
            #scheduler = PolynomialLRDecay(optimizer, max_iters=total_iter, start_lr=args.learning_rate, end_lr=0.00001, power=1)#end_lr=0.00001  
            
            steps_per_epoch = len(train_loader)
            pct_start = 2000 / (args.epochs * steps_per_epoch)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.learning_rate,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                # pct_start=pct_start,
                pct_start=0.3,
            )            
            criteria = {
                "mse": nn.MSELoss(),
                "l1": nn.L1Loss(),
                "huber": nn.HuberLoss(),
            }
            criterion = criteria[args.loss]
            MAE = nn.L1Loss()             
            
            
            
            
            
            for epoch in range(args.epochs):
                model.train()
                running_loss = 0.0
                with tqdm(total=len(train_loader), desc=f"Fold {ii} Epoch {epoch + 1}/{args.epochs}", unit='batch') as pbar:
                    for data in train_loader:  
                        structure,  labels = data
                        
                 
                        structure, labels = structure.to(device),  labels.to(device)                              
                        optimizer.zero_grad()
                        outputs = model(structure).view(-1, 3, 3)
                        outputs = (outputs + outputs.transpose(-1, -2)) / 2  
    
                        outputs = dielectric_to_refractive_index(outputs)
                                  
                        loss = criterion(outputs, labels)
         
                        loss.backward()
                        optimizer.step()
        
                        running_loss += loss.item()
                        pbar.set_postfix({'training_loss': running_loss / (pbar.n + 1)})
                        pbar.update(1)
                        scheduler.step()
        
                    model.eval()
                    running_loss = 0.0
                    label_list = []
                    output_list = []        
                    for data in val_loader:
                        structure, labels = data
                        structure, labels = structure.to(device), labels.to(device)
    
                        outputs = model(structure).detach()
                        outputs = outputs.view(-1, 3, 3)
                        outputs = (outputs + outputs.transpose(-1, -2)) / 2                 
                        outputs = dielectric_to_refractive_index(outputs)  
            
                        output_list.append(outputs.reshape(-1))
            
                        label_list.append(labels.reshape(-1))
            
                    
                    outputs = torch.stack(output_list).reshape(-1)
                    labels = torch.stack(label_list).reshape(-1)
                    mae = abs(outputs - labels).mean(dim=-1).mean()

                    print("Validation mae ", mae)   
        
        
         
            # Get testing data
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_inputs=test_inputs.tolist()
            test_outputs=test_outputs.tolist()
            dataset_test = []
            for i in tqdm(range(len(test_inputs))):   
                temp={'structure':test_inputs[i],'target':test_outputs[i]}
                dataset_test.append(temp) 
            pyg_dataset_test = get_pyg_dataset(dataset_test, args.subtarget, args.reduce_cell)   

            test_loader = DataLoader(
                pyg_dataset_test,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
            )   
            
            mae_list =[]
            frob_list = []
            percen_list = []
            out_list = []
            label_list = []    
            error_eT = [] 
            print(f"Fold {ii} test...")
            for data in tqdm(test_loader):
                structure, labels = data
                structure, labels = structure.to(device),  labels.to(device)

                outputs = model(structure).view(-1,3, 3) # 3 * 3
                outputs = (outputs + outputs.transpose(-1, -2)) / 2  
                outputs = dielectric_to_refractive_index(outputs)           
                outputs = outputs.cpu().detach()
                
                 
                out_list.append(outputs)

            output_tensor = torch.stack(out_list)
            output_tensor=output_tensor.numpy()

            task.record(fold, output_tensor)
    
    # Save your results
    mb.to_file("my_models_benchmark_ic.json.gz")    
        
    scores = mb.matbench_dielectric.scores
    print(scores ) 
 
    return




def main():
    parser = argparse.ArgumentParser(description='Training script')

    # Define command-line arguments
    # training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training and evaluating')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-05, help='weight decay')
    parser.add_argument('--loss', type=str, default='huber', help='mse or l1 or huber')
    parser.add_argument('--model', type=str, default='GoeCTP', help='GoeCTP')
    

    parser.add_argument('--project', type=str, default='test', help='name of project for wandb visualization')
    parser.add_argument('--name', type=str, default='test', help='name of project for storage')
    parser.add_argument('--reduce_cell', type=bool, default=False, help='reduce the cell into irreducible atom sets, not used')
    parser.add_argument('--use_mask', type=bool, default=True, help='symmetry correction module introduced in the paper')
    # dataset parameters
    parser.add_argument('--split_seed', type=int, default=32, help='the random seed of spliting data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='training ratio used in data split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='evaluate ratio used in data split')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test ratio used in data split')
    parser.add_argument('--target', type=str, default='dielectric', help='dielectric')
    parser.add_argument('--subtarget', type=str, default='total', help='total, ionic, or electronic')#choose subtarget-total, ionic, or electronic in DTnet paper

    
    parser.add_argument('--test_augment', type=str, default='None', help='None, XZ_exchange, Xrotate, Yrotate, Zrotate')
    parser.add_argument('--threshold', type=float, default=100., help='threshold to remove samples')
    parser.add_argument('--use_corrected_structure', type=bool, default=False, help='correct input structure or not')
    parser.add_argument('--load_model', type=bool, default=False, help='load pretrained model or not')#False True
    parser.add_argument('--load_preprocessed', type=bool, default=True, help='load previous processed dataset')#False True   first time should be False, later repeat is ture

    args = parser.parse_args()
    '''
    ewald_hyperparams={
      'num_k_x': 1,                              # Frequency cutoff [?^-1]
      'num_k_y': 1 ,                             # Voxel grid resolution [?^-1]
      'num_k_z': 1 ,                             # Gaussian radial basis size (Fourier filter)
      'downprojection_size': 4 ,                 # Size of linear bottleneck layer
      'num_hidden': 3    ,                       # Number of residuals in update function
    }  
    '''
    print('Training settings:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Learning rate: {args.learning_rate}')
    print(args)
    torch.manual_seed(args.split_seed)
    torch.cuda.manual_seed_all(args.split_seed)
    # load the model
    if args.model == "GoeCTP":
        #model = GoeCTP_plus(args,ewald_hyperparams)#,ewald_hyperparams
        model = GoeCTP_plus(args)


    if not os.path.exists('runs/' + args.name):
        # Create the directory
        os.makedirs('runs/' + args.name)
        
    if args.load_model:

        saved_model_path = "./yourpath"   
        state_dict = torch.load(saved_model_path)
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)
 
    

    train(model, args)


if __name__ == "__main__":
    main()
