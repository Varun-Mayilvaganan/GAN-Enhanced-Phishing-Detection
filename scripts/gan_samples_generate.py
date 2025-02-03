import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load the datasets
phishing = pd.read_csv('/content/phishing.csv')
legitimate = pd.read_csv('/content/legitimate.csv')

phishing_data = phishing.drop(columns=['Domain'])
legitimate_data = legitimate.drop(columns=['Domain'])

# Convert data to numpy arrays
data_phishing = phishing_data.to_numpy()
data_legitimate = legitimate_data.to_numpy()

# Set the device to CUDA if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

# Define Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Function to train GAN and generate samples
def train_gan(data, data_columns, output_file, num_epochs=10000, batch_size=64, latent_dim=100):
    data_dim = data.shape[1]
    generator = Generator(latent_dim, data_dim).to(device)
    discriminator = Discriminator(data_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    adversarial_loss = nn.BCELoss()

    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for real_samples in dataloader:
            batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_loss_real = adversarial_loss(discriminator(real_samples), real_labels)
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = generator(z)
            d_loss_fake = adversarial_loss(discriminator(fake_samples), fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim).to(device)
            generated_samples = generator(z)
            g_loss = adversarial_loss(discriminator(generated_samples), real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")
        if epoch == 500:
            print("Trained successfully")
            break

    # Generate new samples
    z = torch.randn(1000, latent_dim).to(device)
    generated_samples = generator(z).detach().cpu().numpy()
    binary_samples = (generated_samples > 0.5).astype(int)
    generated_df = pd.DataFrame(binary_samples, columns=data_columns)
    generated_df.to_csv(output_file, index=False)

# Train GAN for phishing and legitimate data
train_gan(data_phishing, phishing_data.columns, 'generated_phishing_samples.csv')
train_gan(data_legitimate, legitimate_data.columns, 'generated_legitimate_samples.csv')
