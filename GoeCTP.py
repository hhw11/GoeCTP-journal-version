import torch
from torch import nn
from utils import RBFExpansion
from transformer import *
from torch_scatter import scatter
#from ewald_block import *
#from scipy.linalg import polar

import torch.nn.functional as F
import math


def polar_decomposition_torch(A: torch.Tensor):
    # A: (..., N, N)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = torch.matmul(U, Vh)
    P = torch.matmul(Vh.transpose(-2, -1), torch.diag_embed(S)) @ Vh
    return R, P  # R: rotation-like (orthogonal), P: symmetric positive semi-definite



def bond_cosine(r1, r2):
    dot_product = torch.sum(r1 * r2, dim=-1)
    norm1 = torch.norm(r1, dim=-1)
    norm2 = torch.norm(r2, dim=-1)
    norm_product = norm1 * norm2


    is_zero = norm_product == 0
    cosine = dot_product / (norm_product + 1e-8)  
    cosine = torch.clamp(cosine, -1, 1)
    cosine[is_zero] = 1.0

    return cosine

class GoeCTP_plus(nn.Module):
    def __init__(self, args):
        super().__init__()
        embsize=128
        self.backbone=Backbone(args)
        #self.backbone=Backbone2(args)        
        self.fc_out = nn.Sequential(
            nn.Linear(embsize, embsize),
            nn.ELU(),
            nn.Linear(embsize, 9 ),
            #nn.ReLU(),
        )   
                
                
                
    def forward(self, data) -> torch.Tensor:
        outputs,rot=self.backbone(data)
        
        outputs = self.fc_out(outputs)     
        
        outputs=outputs.view(-1,3, 3)
        outputs=torch.bmm(rot,torch.bmm(outputs, rot.transpose(-1, -2)) ) 
        return outputs 


class FourierPositionalEncoding(nn.Module):
    def __init__(self, input_dim, fourier_dim, hidden_dim, output_dim, gamma):
        super().__init__()
        
        # learnable Fourier projection matrix: W_r
        self.W_r = nn.Parameter(
            torch.randn(fourier_dim // 2, input_dim) * (1.0 / gamma)
        )  # shape: [|F|/2, M]

        # Replacing (W1, B1) and (W2, B2) with Linear layers
        self.linear1 = nn.Linear(fourier_dim, hidden_dim)   # corresponds to W1 + B1
        self.linear2 = nn.Linear(hidden_dim, output_dim)    # corresponds to W2 + B2
        #self.linear_proj = nn.Linear(input_dim, fourier_dim // 2, bias=False)

    def forward(self, X):  # X: [N, M]
        # Fourier projection
        X_proj = 2 * math.pi * X @ self.W_r.T  # [N, |F|/2]
        #X_proj = 2 * math.pi * self.linear_proj(X)
        F_features = torch.cat([torch.cos(X_proj), torch.sin(X_proj)], dim=-1)  # [N, |F|]

        # Feed-forward through two Linear layers
        Y = self.linear2(F.gelu(self.linear1(F_features)))  # [N, D]

        return Y
        
class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        embsize = 128
        
        

            
                                      
        self.atom_embedding = nn.Linear(
            92, embsize
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=512,
            ),
            nn.Linear(512, embsize),
            nn.Softplus(),
        )
        '''     
        self.pos_encoder =FourierPositionalEncoding(input_dim=3, fourier_dim=embsize //8, hidden_dim=embsize //4, output_dim=embsize//2, gamma=1)
        '''         
        ###self.pos_encoder =FourierPositionalEncoding(input_dim=3, fourier_dim=embsize //4, hidden_dim=embsize //2, output_dim=embsize, gamma=1)        
        
        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=512,
            ),
            nn.Linear(512,embsize),
            nn.Softplus(),
        )
        
        '''
        self.rotnet =     nn.ModuleList(
            [
                Rotnet_global(in_channels=embsize, out_channels=embsize,heads=1,  edge_dim=embsize) 
                for _ in range(1)
            ]
        )  
        '''

        self.att_layers = nn.ModuleList(
            [
           
                ComformerConv(in_channels=embsize, out_channels=embsize, heads=1, edge_dim=embsize)
                #EGNN(in_channels=embsize, out_channels=embsize, heads=1, edge_dim=embsize)
                for _ in range(4)
            ]
        )
        self.edge_update_layer =     nn.ModuleList(
            [
                ComformerConv_edge(in_channels=embsize, out_channels=embsize, heads=1, edge_dim=embsize)
                for _ in range(1)
            ]
        )


                
    def forward(self, data) -> torch.Tensor:


        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        edge_features = self.rbf(edge_feat)
        
        batch = data.batch
        batch_size = int(batch.max()) + 1
        
        lattice=data.cell
        UT, PT = polar_decomposition_torch(lattice.transpose(-2, -1))
        #UT, PT = polar(lattice_as_numpy.T)
        U=UT.transpose(-2, -1)
        #P=PT.transpose(-2, -1) 
        
        #rotlist = self.rotnet[0](data, node_features, data.edge_index, edge_features)     
        #rotlist= torch.matmul(UT, rotlist)
        rotlist=UT

 
        pos=data.pos
        rotlisttemp=rotlist[batch]
        pos = torch.bmm(pos.unsqueeze(1), rotlisttemp).squeeze(1)
        posshape=pos.shape
        pos_feat = -0.75 / torch.norm(data.pos, dim=1)
   




        edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1) # [num_edges, 3]
        edge_nei_angle = bond_cosine(data.edge_nei, data.edge_attr.unsqueeze(1).repeat(1, 3, 1)) # [num_edges, 3, 3] -> [num_edges, 3]
        num_edge = edge_feat.shape[0]

        edge_nei_len = self.rbf(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)
        


        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)

        edge_features=self.edge_update_layer[0](edge_features, edge_nei_len, edge_nei_angle)



        node_features = self.att_layers[1](node_features, data.edge_index, edge_features) 

           
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        #node_features = self.att_layers[4](node_features, data.edge_index, edge_features)
        #node_features = self.att_layers[5](node_features, data.edge_index, edge_features)         
   
        
        # global mean pooling for high rotation order features
        outputs = scatter(node_features, data.batch, dim=0, reduce="mean")

        return outputs,rotlist        

















'''
       
class Backbone_ew(nn.Module):
    def __init__(self, args,ewald_hyperparams=None):
        super().__init__()
        embsize = 128
        
        
        self.use_ewald = ewald_hyperparams is not None
        
        if self.use_ewald:
            #if self.use_pbc:
                # Integer values to define box of k-lattice indices
            self.num_k_x = ewald_hyperparams["num_k_x"]
            self.num_k_y = ewald_hyperparams["num_k_y"]
            self.num_k_z = ewald_hyperparams["num_k_z"]
            self.delta_k = None  
                
            self.downprojection_size = ewald_hyperparams["downprojection_size"]
            # Number of residuals in update function
            self.num_hidden = ewald_hyperparams["num_hidden"]
            # Whether to detach Ewald blocks from the computational graph
            # (recommended for DimeNet++, as the Ewald embeddings do not
            # mix with the remaining hidden state of the model prior to output
            # and the Ewald blocks barely contribute to the force predictions)
            self.detach_ewald = ewald_hyperparams.get("detach_ewald", True)  
                          
        if self.use_ewald:
            # Initialize k-space structure
            #if self.use_pbc:
                # Get the reciprocal lattice indices of included k-vectors
            (
                self.k_index_product_set,
                self.num_k_degrees_of_freedom,
            ) = get_k_index_product_set(
                self.num_k_x,
                self.num_k_y,
                self.num_k_z,
            )
            self.k_rbf_values = None
            self.delta_k = None          
            self.down = Dense(
                self.num_k_degrees_of_freedom,
                self.downprojection_size,
                activation=None,
                bias=False,
            )  
            
            self.ewald_blocks = torch.nn.ModuleList(
                [
                    EwaldBlock(
                        self.down,
                        embsize,  # Embedding size of short-range GNN
                        self.downprojection_size,
                        self.num_hidden,  # Number of residuals in update function
                        activation="silu",
                        use_pbc=True,
                        delta_k=self.delta_k,
                        k_rbf_values=self.k_rbf_values,
                    )
                    for i in range(4)
                ]
            )            
            
            
            
                                      
        self.atom_embedding = nn.Linear(
            92, embsize
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=512,
            ),
            nn.Linear(512, embsize),
            nn.Softplus(),
        )
        self.rbf2 = nn.Sequential(
            RBFExpansion(
                vmin=-40.0,
                vmax=0.0,
                bins=512,
            ),
            nn.Linear(512, embsize),
            nn.Softplus(),
        )
                
        self.pos_encoder =FourierPositionalEncoding(input_dim=3, fourier_dim=embsize, hidden_dim=embsize, output_dim=embsize, gamma=1)
        
        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=512,
            ),
            nn.Linear(512,embsize),
            nn.Softplus(),
        )
        
        
        self.rotnet =     nn.ModuleList(
            [
                Rotnet_global(in_channels=embsize, out_channels=embsize,heads=1,  edge_dim=embsize) 
                for _ in range(1)
            ]
        )  


        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=embsize, out_channels=embsize, heads=1, edge_dim=embsize)
                for _ in range(4)
            ]
        )
        self.edge_update_layer =     nn.ModuleList(
            [
                ComformerConv_edge(in_channels=embsize, out_channels=embsize, heads=1, edge_dim=embsize)
                for _ in range(1)
            ]
        )

        self.node_update_layer =     nn.ModuleList(
            [
                #ComformerConv_pos(in_channels=embsize, out_channels=embsize, heads=1)
                ComformerConv_pos2(in_channels=embsize, out_channels=embsize, heads=1)
                for _ in range(1)
            ]
        )   
    
        self.skip_connection_factor = (
            1.0 + float(self.use_ewald)
        ) ** (-0.5)
                
    def forward(self, data) -> torch.Tensor:


        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        # edge_feat = torch.norm(data.edge_attr, dim=1)
        edge_features = self.rbf(edge_feat)

        edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1) # [num_edges, 3]
        edge_nei_angle = bond_cosine(data.edge_nei, data.edge_attr.unsqueeze(1).repeat(1, 3, 1)) # [num_edges, 3, 3] -> [num_edges, 3]
        num_edge = edge_feat.shape[0]

        edge_nei_len = self.rbf(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)
        
        
        batch = data.batch
        batch_size = int(batch.max()) + 1
        
        lattice=data.cell
        UT, PT = polar_decomposition_torch(lattice.transpose(-2, -1))
        #UT, PT = polar(lattice_as_numpy.T)
        U=UT.transpose(-2, -1)
        #P=PT.transpose(-2, -1) 
        
        rotlist = self.rotnet[0](data, node_features, data.edge_index, edge_features)  
         
         
        rotlist= torch.matmul(UT, rotlist)
 
        pos=data.pos
        rotlisttemp=rotlist[batch]
        pos = torch.bmm(pos.unsqueeze(1), rotlisttemp).squeeze(1)
        posshape=pos.shape

        pos_feat = -0.75 / torch.norm(data.pos, dim=1)
        # edge_feat = torch.norm(data.edge_attr, dim=1)
        pos_emb = self.rbf2(pos_feat)
        latticetemp=lattice[batch]
        pos_nei_angle = bond_cosine(latticetemp, data.pos.unsqueeze(1).repeat(1, 3, 1)) 
        
        num_pos = pos_feat.shape[0]


        pos_nei_angle = self.rbf_angle(pos_nei_angle.reshape(-1)).reshape(num_pos, 3, -1)        
        
        #pos_emb=self.pos_encoder(pos)
        #pos_nei_len = self.rbf(pos.reshape(-1)).reshape(posshape)


        if self.use_ewald:        
            # Compute reciprocal lattice basis of structure
            k_cell, _ = x_to_k_cell(data.cell)
            # Translate lattice indices to k-vectors
            k_grid = torch.matmul(
                self.k_index_product_set.to(batch.device), k_cell
            )        
            dot = None  # These will be computed in first Ewald block and then passed
            sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
            #pos_detach = pos.detach() if self.detach_ewald else pos
        

        node_features=self.node_update_layer[0](node_features, pos_emb,pos_nei_angle) 
        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
     
      
        #node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        #edge_features=self.edge_update_layer[0](edge_features, edge_nei_len, edge_nei_angle)
        #node_features=self.node_update_layer[0](node_features, pos_emb)     
       
           
        if self.use_ewald:
            h_ewald, dot, sinc_damping = self.ewald_blocks[0](
                node_features,
                pos,
                k_grid,
                batch_size,
                batch,
                dot,
                sinc_damping,
            ) 
            node_features = self.skip_connection_factor * (node_features + h_ewald)       

        #h = self.skip_connection_factor * (h + h_ewald)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features) 
        #node_features = self.equi_update2(data, node_features, data.edge_index, edge_features) 
           
         
        if self.use_ewald:
            h_ewald, dot, sinc_damping = self.ewald_blocks[1](
                node_features,
                pos,
                k_grid,
                batch_size,
                batch,
                dot,
                sinc_damping,
            ) 
            node_features = self.skip_connection_factor * (node_features + h_ewald)             
              
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        if self.use_ewald:
            h_ewald, dot, sinc_damping = self.ewald_blocks[2](
                node_features,
                pos,
                k_grid,
                batch_size,
                batch,
                dot,
                sinc_damping,
            ) 
            node_features = self.skip_connection_factor * (node_features + h_ewald)         
        #node_features = self.equi_update3(data, node_features, data.edge_index, edge_features)         
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        if self.use_ewald:
            h_ewald, dot, sinc_damping = self.ewald_blocks[3](
                node_features,
                pos,
                k_grid,
                batch_size,
                batch,
                dot,
                sinc_damping,
            ) 
            node_features = self.skip_connection_factor * (node_features + h_ewald)         
        #node_features = self.equi_update4(data, node_features, data.edge_index, edge_features) 
        
        # global mean pooling for high rotation order features
        outputs = scatter(node_features, data.batch, dim=0, reduce="mean")

        return outputs,rotlist            
'''