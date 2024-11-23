# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split


# ##Load and Preprocess the Dataset
# # Load the dataset
# df = pd.read_csv('creditcard.csv')

# # Separate the minority class (class 1)
# minority_df = df[df['Class'] == 1]

# # Drop the 'Class' column for training
# X_minority = minority_df.drop(columns=['Class'])

# # Standardize the data
# scaler = StandardScaler()
# X_minority_scaled = scaler.fit_transform(X_minority)

# # Convert to PyTorch tensor
# X_minority_tensor = torch.tensor(X_minority_scaled, dtype=torch.float32).cuda()


# ##Define the Transformer Model
# class SmallTransformerDiffusionModel(nn.Module):
#     def __init__(self, input_dim, emb_dim=64, n_heads=2, n_layers=2, forward_expansion=2, dropout=0.1):
#         super(SmallTransformerDiffusionModel, self).__init__()
        
#         self.emb_dim = emb_dim
#         self.input_projection = nn.Linear(input_dim, emb_dim)  # Added input projection to match dimensions
#         self.pos_embedding = nn.Embedding(1000, emb_dim)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=emb_dim, 
#             nhead=n_heads, 
#             dim_feedforward=forward_expansion * emb_dim,
#             dropout=dropout
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.fc1 = nn.Linear(emb_dim, emb_dim)
#         self.fc2 = nn.Linear(emb_dim, input_dim)
    
#     def forward(self, x):
#         seq_len, batch_size, _ = x.shape
        
#         # Project input to match embedding dimension
#         x = self.input_projection(x)
        
#         positions = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, batch_size).cuda()
#         x = x + self.pos_embedding(positions)
#         encoded_x = self.encoder(x)
#         output = self.fc1(encoded_x)
#         output = torch.relu(output)
#         output = self.fc2(output)
        
#         return output


# ##Define the Diffusion Process
# class DiffusionProcess:
#     def __init__(self, model, timesteps):
#         self.model = model
#         self.timesteps = timesteps
    
#     def forward_diffusion(self, x):
#         noise = torch.randn_like(x).cuda()
#         return x + noise
    
#     def reverse_diffusion(self, x):
#         for t in reversed(range(self.timesteps)):
#             x = self.model(x)
#             x = F.relu(x)
#         return x


# ##5. Train the Model on the Minority Class
# # Prepare data loader
# batch_size = 32
# train_loader = torch.utils.data.DataLoader(X_minority_tensor, batch_size=batch_size, shuffle=True)

# # Initialize the model and optimizer
# input_dim = X_minority_tensor.shape[1]
# model = SmallTransformerDiffusionModel(input_dim=input_dim).cuda()
# diffusion = DiffusionProcess(model=model, timesteps=100)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     for batch in train_loader:
#         noisy_data = diffusion.forward_diffusion(batch.unsqueeze(0))
#         reconstructed_data = diffusion.reverse_diffusion(noisy_data)
#         loss = F.mse_loss(reconstructed_data, batch.unsqueeze(0))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# ##6. Generate 500 Synthetic Samples
# # Generate 500 synthetic samples
# seq_len = X_minority_tensor.shape[0]
# noise = torch.randn((seq_len, batch_size, input_dim)).cuda()
# synthetic_data = diffusion.reverse_diffusion(noise)
# synthetic_data = synthetic_data.squeeze(0)[:500].detach().cpu().numpy()

# # Convert to DataFrame and inverse transform
# synthetic_df = pd.DataFrame(scaler.inverse_transform(synthetic_data), columns=X_minority.columns)


# #7. Plot the Original and Synthetic Data Side by Side
# # Plotting the original and synthetic data
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Plot original minority class
# axes[0].scatter(X_minority_scaled[:, 0], X_minority_scaled[:, 1], alpha=0.5)
# axes[0].set_title("Original Minority Class")

# # Plot synthetic data
# axes[1].scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.5, color='red')
# axes[1].set_title("Synthetic Minority Class")

# plt.show()

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set environment variable for CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Separate the minority class (class 1)
minority_df = df[df['Class'] == 1]

# Drop the 'Class' column for training
X_minority = minority_df.drop(columns=['Class'])

# Standardize the data
scaler = StandardScaler()
X_minority_scaled = scaler.fit_transform(X_minority)

# Convert to PyTorch tensor
X_minority_tensor = torch.tensor(X_minority_scaled, dtype=torch.float32).cuda()

# Define LoRA Layer
class LoRALayer(nn.Module):
    def __init__(self, original_dim, rank=4):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.original_dim = original_dim

        # Low-rank matrix decomposition
        self.A = nn.Parameter(torch.randn(original_dim, rank))  # Low-rank matrix A
        self.B = nn.Parameter(torch.randn(rank, original_dim))  # Low-rank matrix B

    def forward(self, x):
        return x + (x @ self.A @ self.B)  # xW = x(A @ B)

# Define LoRA-augmented Attention
class LoRAAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, rank=4):
        super(LoRAAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.rank = rank

        # Original self-attention
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads)

        # LoRA applied to attention weights
        self.q_lora = LoRALayer(emb_dim, rank=rank)
        self.k_lora = LoRALayer(emb_dim, rank=rank)
        self.v_lora = LoRALayer(emb_dim, rank=rank)

    def forward(self, x):
        # Apply LoRA to query, key, and value matrices
        q = self.q_lora(x)
        k = self.k_lora(x)
        v = self.v_lora(x)

        attn_output, _ = self.attention(q, k, v)
        return attn_output

# Define LoRA Transformer Encoder Layer
class LoRATransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, forward_expansion=1, dropout=0.1, rank=4):
        super(LoRATransformerEncoderLayer, self).__init__()
        self.attention = LoRAAttention(emb_dim, n_heads, rank=rank)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, forward_expansion * emb_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_dim, emb_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))
        return x

# Define the LoRA Transformer model
class LoRATransformer(nn.Module):
    def __init__(self, input_dim, emb_dim=16, n_heads=1, n_layers=1, forward_expansion=1, dropout=0.1, rank=4):
        super(LoRATransformer, self).__init__()
        self.emb_dim = emb_dim
        self.input_projection = nn.Linear(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(1000, emb_dim)
        
        # Stack of LoRA Transformer Encoder layers
        self.layers = nn.ModuleList([
            LoRATransformerEncoderLayer(emb_dim, n_heads, forward_expansion, dropout, rank) 
            for _ in range(n_layers)
        ])

        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, input_dim)
    
    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        x = self.input_projection(x)
        positions = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, batch_size).cuda()
        x = x + self.pos_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        output = self.fc1(x)
        output = torch.relu(output)
        output = self.fc2(output)
        return output

# Define the Diffusion Process with the LoRA Transformer model
class DiffusionProcess:
    def __init__(self, model, timesteps):
        self.model = model
        self.timesteps = timesteps
    
    def forward_diffusion(self, x):
        noise = torch.randn_like(x).cuda()
        return x + noise
    
    def reverse_diffusion(self, x):
        for t in reversed(range(self.timesteps)):
            x = self.model(x)
            x = F.relu(x)
        return x

# Prepare data loader
batch_size = 4  # Small batch size
train_loader = torch.utils.data.DataLoader(X_minority_tensor, batch_size=batch_size, shuffle=True)

# Initialize the model and optimizer
input_dim = X_minority_tensor.shape[1]
model = LoRATransformer(input_dim=input_dim).cuda()
diffusion = DiffusionProcess(model=model, timesteps=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Mixed Precision Training Setup
scaler = torch.cuda.amp.GradScaler()

# Gradient accumulation
accumulation_steps = 2

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        with torch.cuda.amp.autocast():  # Enable mixed precision
            noisy_data = diffusion.forward_diffusion(batch.unsqueeze(0))
            reconstructed_data = diffusion.reverse_diffusion(noisy_data)
            loss = F.mse_loss(reconstructed_data, batch.unsqueeze(0))

        scaler.scale(loss).backward()  # Scale the loss

        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scaler.update()  # Update the scaler
            torch.cuda.empty_cache()  # Clear cache after weight update
        
    # Print and clear cache after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    torch.cuda.empty_cache()  # Clear cache after printing loss

# Generate 500 synthetic samples
seq_len = X_minority_tensor.shape[0]
noise = torch.randn((seq_len, batch_size, input_dim)).cuda()
with torch.no_grad():  # Disable gradient calculation during inference
    synthetic_data = diffusion.reverse_diffusion(noise)
synthetic_data = synthetic_data.squeeze(0)[:500].detach().cpu().numpy()

# Convert to DataFrame and inverse transform
synthetic_df = pd.DataFrame(scaler.inverse_transform(synthetic_data), columns=X_minority.columns)

# Save the synthetic data to a CSV file
synthetic_df.to_csv('synthetic_creditcard_data.csv', index=False)

# Plotting the original and synthetic data
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(X_minority_scaled[:, 0], X_minority_scaled[:, 1], alpha=0.5)
axes[0].set_title("Original Minority Class")
axes[1].scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.5, color='red')
axes[1].set_title("Synthetic Minority Class")
plt.show()
