import torch
import torch.nn as nn
import math

# create a class for embedding and a class for embedding position
# note that each class gets passed the nn.module base class 
#class input_embedding(nn.Module):: This defines a custom class called input_embedding that extends (inherits from) nn.Module,
# which is the base class for all neural network layers in PyTorch. 
# By inheriting from nn.Module, we can make use of PyTorch's powerful features like backpropagation, parameter management, etc.

class input_embedding(nn.Module):
    #this class contains two attributes: d_model and vocab_size
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
 #we multiply the embedding by the square root of the d_model& return it through the forward method 
 #this is the same as the formula in the transformer paper      
    def forward(self.x):
        return self.embedding(x)*math.sqrt(self.d_model)
    