import tqdm
import gc
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import process_GQ as pce


def _make_mlp_layers(num_units, dropout_rate=0.5):
    layers = nn.ModuleList()
    for in_features, out_features in zip(num_units, num_units[1:]):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Dropout(p=dropout_rate))
    return layers

class DeapStack(nn.Module):
    ''' Simple MLP body. '''
    def __init__(self,  n_bins, n_cats, n_nums, cards, in_features, hidden_size, bottleneck_size, num_layers, dropout_rate=0.5):
        super().__init__()       
        encoder_layers = num_layers >> 1
        decoder_layers = num_layers - encoder_layers - 1
        self.encoders = _make_mlp_layers([in_features] + [hidden_size] * encoder_layers, dropout_rate)
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.decoders = _make_mlp_layers([bottleneck_size] + [hidden_size] * decoder_layers, dropout_rate)

        self.n_bins = n_bins
        self.n_cats = n_cats       
        self.n_nums = n_nums

    def forward_pass(self, x):
        for layer in self.encoders:
            x = layer(x)
        x = self.bottleneck(x)
        for layer in self.decoders:
            x = layer(x)
        return x

    def featurize(self, x):
        for layer in self.encoders:
            x = layer(x)
        return x

    def decoder(self, latent_feature, num_min_values, num_max_values):
        decoded_outputs = dict()

        for layer in self.decoders:
            x = F.relu(layer(latent_feature))
        last_hidden_layer = x
        
        if self.bins_linear:
            decoded_outputs['bins'] = self.bins_linear(last_hidden_layer)

        if self.cats_linears:
            decoded_outputs['cats'] = [linear(last_hidden_layer) for linear in self.cats_linears]            
            
        if self.nums_linear:
            d_before_threshold = self.nums_linear(last_hidden_layer)
            decoded_outputs['nums'] = d_before_threshold
            
            for col in range(len(num_min_values)):
                decoded_outputs['nums'][:,col] = torch.where(d_before_threshold[:,col] < num_min_values[col], num_min_values[col], d_before_threshold[:,col])     
                decoded_outputs['nums'][:,col] = torch.where(d_before_threshold[:,col] > num_max_values[col], num_max_values[col], d_before_threshold[:,col]) 
        return decoded_outputs
def auto_loss(inputs, reconstruction, n_bins, n_nums, n_cats, cards):
    """ Calculating the loss for DAE network.
        BCE for masks and reconstruction of binary inputs.
        CE for categoricals.
        MSE for numericals.
        reconstruction loss is weighted average of mean reduction of loss per datatype.
        mask loss is mean reduced.
        final loss is weighted sum of reconstruction loss and mask loss.
    """
    bins = inputs[:,0:n_bins]
    cats = inputs[:,n_bins:n_bins+n_cats]
    nums = inputs[:,n_bins+n_cats:n_bins+n_cats+n_nums]
    
    reconstruction_losses = dict()
        
    if 'bins' in reconstruction:
        reconstruction_losses['bins'] = F.binary_cross_entropy_with_logits(reconstruction['bins'], bins)

    if 'cats' in reconstruction:
        cats_losses = []
        for i in range(len(reconstruction['cats'])):
            cats_losses.append(F.cross_entropy(reconstruction['cats'][i], cats[:, i].long()))
        reconstruction_losses['cats'] = torch.stack(cats_losses).mean()        
        
    if 'nums' in reconstruction:
        reconstruction_losses['nums'] = F.mse_loss(reconstruction['nums'], nums)
        
    reconstruction_loss = torch.stack(list(reconstruction_losses.values())).mean()
    return reconstruction_loss

def train_autoencoder(df, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold):
    parser = pce.DataFrameParser().fit(df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype('float32'))

    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']; n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']; cards = datatype_info['cards']

    DS = DeapStack(n_bins, n_cats, n_nums, cards, data.shape[1], hidden_size=hidden_size, bottleneck_size=data.shape[1], num_layers=num_layers)

    optimizer = AdamW(DS.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(data)//batch_size, epochs=n_epochs)

    tqdm_epoch = tqdm.notebook.trange(n_epochs)

    losses = []
    all_indices = list(range(data.shape[0]))

    for epoch in tqdm_epoch:
        batch_indices = random.sample(all_indices, batch_size)
        batch_data = data[batch_indices,:]

        output = DS.forward_pass(batch_data)
        l2_loss = auto_loss(batch_data, output, n_bins, n_nums, n_cats, cards)
        optimizer.zero_grad()
        l2_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

        # Print the training loss over the epoch.
        losses.append(l2_loss.item())
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(l2_loss.item()))
    
    num_min_values, _ = torch.min(data[:,n_bins+n_cats:n_bins+n_cats+n_nums], dim=0)
    num_max_values, _ = torch.max(data[:,n_bins+n_cats:n_bins+n_cats+n_nums], dim=0)

    latent_features = DS.featurize(data)
    output = DS.decoder(latent_features, num_min_values, num_max_values)

    return (DS.decoder, latent_features, num_min_values, num_max_values)