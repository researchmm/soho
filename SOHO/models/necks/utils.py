import torch
import torch.nn as nn
import torch.nn.functional as F



class SOHO_direct_VD(nn.Module):
    def __init__(self,):
        super(SOHO_direct_VD, self).__init__()
        
        self.codebook_indices = None


    def forward(self,inputs):

        x = inputs
        encoding_indices = torch.argmax(x, dim=1).unsqueeze(1)
        return x,encoding_indices
