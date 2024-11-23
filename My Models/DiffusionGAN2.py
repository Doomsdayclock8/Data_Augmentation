import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the dataset (replace with your own tabular data)
df = pd.read_csv('creditcard.csv')
data = torch.tensor(df.values, dtype=torch.float32).to(device)

# Hyperparameters
input_dim = data.shape[1]
latent_dim = 64
num_epochs = 100
batch_size = 32
num_steps = 10  # Number of diffusion steps
lr = 1e-4

# Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, input_dim)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(x))

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Diffusion Process
def forward_diffusion(x, t):
    noise = torch.randn_like(x) * (1.0 - t / num_steps)  # Noise level decreases with time
    return x + noise

def reverse_diffusion(x, t):
    # A very simple reverse process (placeholder, can be improved)
    return x * (1.0 - t / num_steps)

# Initialize models and optimizers
generator = Generator(input_dim, latent_dim).to(device)
discriminator = Discriminator(input_dim).to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(data), batch_size):
        real_data = data[i:i+batch_size]
        
        # Sample noise and generate fake data
        z = torch.randn(real_data.size(0), latent_dim).to(device)
        fake_data = generator(z)

        # Diffusion Process for real and fake data
        diffused_real = forward_diffusion(real_data, num_steps)
        diffused_fake = forward_diffusion(fake_data.detach(), num_steps)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = nn.BCELoss()(discriminator(diffused_real), torch.ones(real_data.size(0), 1).to(device))
        fake_loss = nn.BCELoss()(discriminator(diffused_fake), torch.zeros(real_data.size(0), 1).to(device))
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        generated_diffused = forward_diffusion(fake_data, num_steps)
        loss_G = nn.BCELoss()(discriminator(generated_diffused), torch.ones(real_data.size(0), 1).to(device))
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

# Generate synthetic samples
with torch.no_grad():
    z = torch.randn(1000, latent_dim).to(device)  # Generate 1000 synthetic samples
    synthetic_data = generator(z)
    synthetic_data = synthetic_data.cpu().numpy()

# Convert to DataFrame and save
synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
synthetic_df.to_csv('synthetic_tabular_data.csv', index=False)
