from typing import List, Optional, Tuple, Union, Dict
import json
from monty.json import MontyDecoder
import torch
from ase import Atoms

from torch.utils.data import Dataset
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
import torch
import pickle as pk
import spglib
import numpy as np
from tqdm import tqdm
from scipy.linalg import polar

def get_symmetry_dataset(structure, symprec=1e-5):
    """
    Get space group for a pymatgen Structure object.

    Parameters:
    - structure: pymatgen Structure object
    - symprec: float, the symmetry precision for determining the space group

    Returns:
    - symmetry: dict
    """
    # Convert pymatgen structure to tuple format suitable for spglib
    lattice = structure.lattice.matrix
    positions = structure.frac_coords
    atomic_numbers = structure.atomic_numbers

    cell = (lattice, positions, atomic_numbers)
    # Determine space group
    symmetry = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return symmetry


def rm_duplicates(vectors):
    vecs = vectors.reshape(-1, 9)
    seen = set()
    duplicates = set()
    
    for i in range(vecs.shape[0]):
        vector = vecs[i]
        vt = tuple(vector)  # Convert to tuple for hashability
        if vt in seen:
            duplicates.add(vt)
        else:
            seen.add(vt)
    seen = list(seen)
    vector_list = np.array([list(vector) for vector in seen]).reshape(-1, 3, 3)

    return vector_list

def get_dataset(
    dataset_name="dielectric",
    subtarget='total',
    symprec=1e-5, # Euclidean distance tolerance to determine the space group operations
    use_corrected_structure=False,
    load_preprocessed=False,
):
 
    if load_preprocessed:
        with open(f"./data/preprocessed_{dataset_name}_{subtarget}_dataset.pkl", 'rb') as f:
            dataset = pk.load(f)
            dat = []
            f_norm=[]
            print(len(dataset))
            for i in tqdm(range(len(dataset))):

                dataset[i]['p_input'] = {}


                dataset[i]['p_input']['structure'] = dataset[i]['structure']
                

                f_norm.append((torch.tensor(dataset[i]['target']) ** 2).sum() ** 0.5) 
            print("dataset fnorm mean", torch.tensor(f_norm).mean(), "std", torch.tensor(f_norm).std())

            cubic_cnt = 0
            hexa_cnt = 0
            tetr_cnt = 0
            orth_cnt = 0
            mono_cnt = 0
            tric_cnt = 0
            for i in tqdm(range(len(dataset))):
                space_g = dataset[i]['sym_dataset']['number']
                if space_g >= 195:
                    cubic_cnt += 1
                elif space_g >= 143:
                    hexa_cnt += 1
                elif space_g >= 75:
                    tetr_cnt += 1
                elif space_g >= 16:
                    orth_cnt += 1
                elif space_g >= 3:
                    mono_cnt += 1
                else:
                    tric_cnt += 1
            print("cubic_cnt ", cubic_cnt, "hexa_cnt ", hexa_cnt, "tetr_cnt ", tetr_cnt, "orth_cnt ", orth_cnt, "mono_cnt ", mono_cnt, "tric_cnt ", tric_cnt)
            # dataset = dat
        return dataset
        
    with open("./data/mp_dielectric.json", "r") as f:
        struct_info = json.load(f, cls=MontyDecoder)

    print(len(struct_info))
    dataset = []
    print('load original data................')    
    for key, value in struct_info.items():

        temp={'structure':value['structure'],'target':value[subtarget]}
        dataset.append(temp)


    for i in tqdm(range(len(dataset))):  
    
    
    
        if use_corrected_structure:
            # remove the rotation transformation
            structure=dataset[i]['structure']            
            sym_dataset = get_symmetry_dataset(structure, symprec)            
            Rot = np.array(sym_dataset['std_rotation_matrix'])
            target_tmp = np.array(dataset[i]['target'])
            dataset[i]['target'] = np.dot(Rot, np.dot(target_tmp, Rot.T))

            dataset[i]['structure'] = Structure(lattice=sym_dataset['std_lattice'], species=sym_dataset['std_types'], coords=sym_dataset['std_positions'])   
                
        structure=dataset[i]['structure'] 
        
        
        sym_dataset = get_symmetry_dataset(structure, symprec)

          
        rots = np.array(sym_dataset['rotations'])
        rots = rm_duplicates(rots)
        Lat = structure.lattice.matrix.T
        L_inv = np.linalg.inv(Lat)
      

        tmp_rot = np.matmul(Lat, np.matmul(rots, L_inv)) 
              
        cart_coords = structure.cart_coords
     
        equivalent_atoms = sym_dataset["equivalent_atoms"] 

        lattice = np.array(structure.lattice.matrix.T) 
      
    
        
        dataset[i]['sym_dataset'] =sym_dataset

        dataset[i]['p_input'] = {}
        

        dataset[i]['p_input']['structure'] = dataset[i]['structure']
        

    with open(f"./data/preprocessed_{dataset_name}_{subtarget}_dataset.pkl", 'wb') as f:
        pk.dump(dataset, f)        

    return dataset    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    