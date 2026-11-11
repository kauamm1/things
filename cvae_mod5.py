import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from random import random

# cuda setup
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': use_gpu} 

# hyper params
batch_size = 64
latent_size = 6
epochs = 50


class TwoDigitMNIST(Dataset):
    def __init__(self, root='./data', train=True, download=True):
        # Load the base MNIST dataset
        self.mnist = datasets.MNIST(
            root=root, 
            train=train, 
            download=download, 
            transform=transforms.ToTensor()
        )
        self.length = len(self.mnist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Get the first digit (using the provided index)
        img1, label1 = self.mnist[idx]
        
        # 2. Randomly select an index for the second digit
        idx2 = torch.randint(0, self.length, (1,)).item()
        img2, label2 = self.mnist[idx2]
        
        # 3. Concatenate the images horizontally (dim=2 for HxW after channel)
        # img1 and img2 are shape [1, 28, 28], resulting combined_img is [1, 28, 56]
        combined_img = torch.cat((img1, img2), dim=2)
        
        # 4. Create the combined label (the sum of the digits)
        combined_label = label1 + label2
        return combined_img, combined_label


two_digit_dataset = TwoDigitMNIST(train=True)
dataloader = DataLoader(
    two_digit_dataset, 
    batch_size=64, 
    shuffle=True,
    num_workers=4, 
    pin_memory=use_gpu 
)
two_digit_dataset_test = TwoDigitMNIST(train=False)
test_dataloader = DataLoader(
    two_digit_dataset_test, 
    batch_size=64, 
    shuffle=True,
    num_workers=4, 
    pin_memory=use_gpu 
)

class ProbLogWMC(nn.Module):
    """
    Native PyTorch implementation of ProbLog's logical inference for addition.
    Mathematically computes the exact Weighted Model Counting (WMC) for all proof trees.
    """
    def __init__(self):
        super(ProbLogWMC, self).__init__()

    def forward(self, d1_prob, d2_prob):
        batch_size = d1_prob.size(0)
        # 19 possible sums (0 through 18)
        sum_probs = torch.zeros(batch_size, 19, device=d1_prob.device)
        
        # Logical rule: sum_result(i + j) :- digit1(i), digit2(j).
        for i in range(10):
            for j in range(10):
                # The probability of the sum is the product of the independent digit probabilities
                sum_probs[:, i + j] += d1_prob[:, i] * d2_prob[:, j]
                
        return sum_probs

sum_module = ProbLogWMC().to(device)

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

class LabelPredictor(nn.Module):
    def __init__(self, feature_size=1568, class_size=100):
        super(LabelPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
        )
        self.digit1_head = nn.Linear(256, 10)
        self.digit2_head = nn.Linear(256, 10)

    def forward(self, x):
        features = self.net(x.view(x.size(0), -1))
        return self.digit1_head(features), self.digit2_head(features)

class ConditionPrior(nn.Module):
    def __init__(self, class_size, latent_size):
        super(ConditionPrior, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(class_size, 256),
            nn.ELU(),
            nn.Linear(256, latent_size)
        )

    def forward(self, c):
        return self.net(c)

class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1 = nn.Linear(feature_size + class_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc31 = nn.Linear(512, latent_size)
        self.fc32 = nn.Linear(512, latent_size)

        # decode
        self.fc4 = nn.Linear(latent_size + class_size, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        h2 = self.elu(self.fc2(h1))
        z_mu = self.fc31(h2)
        z_var = self.fc32(h2)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc4(inputs))
        h4 = self.elu(self.fc5(h3))
        return self.sigmoid(self.fc6(h4))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*56), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# create a CVAE model
model = CVAE(28*56, latent_size, 100).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# create the conditional prior
prior_net = ConditionPrior(class_size=100, latent_size=latent_size).to(device)
prior_optimizer = optim.Adam(prior_net.parameters(), lr=1e-3)

# create the label predictor
classifier = LabelPredictor(feature_size=28*56, class_size=100).to(device)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
nll_loss_fn = nn.NLLLoss()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1568), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


def train(epoch):
    model.train()
    prior_net.train()
    classifier.train()
    train_loss = 0
    prior_train_loss = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)

        # 1. Perception Layer (Classifier)
        d1_logits, d2_logits = classifier(data)
        d1_prob = F.softmax(d1_logits, dim=1)
        d2_prob = F.softmax(d2_logits, dim=1)

        # 2. Symbolic Engine Routing (Soft)
        sum_probs = sum_module(d1_prob, d2_prob)

        # 3. cVAE / Prior Network Routing (Hard Joint Condition)
        tau = max(1.0, 2.0 * (1.0 - (epoch / epochs)))
        d1_gumbel = F.gumbel_softmax(d1_logits, tau=tau, hard=True)
        d2_gumbel = F.gumbel_softmax(d2_logits, tau=tau, hard=True)
        c_pred = torch.bmm(d1_gumbel.unsqueeze(2), d2_gumbel.unsqueeze(1)).view(-1, 100)

        recon_batch, mu, logvar = model(data, c_pred)

        optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        prior_optimizer.zero_grad()

        # 1. Entropy Regularization (Decaying)
        d1_entropy = -torch.sum(d1_prob * torch.log(d1_prob + 1e-7), dim=1).mean()
        d2_entropy = -torch.sum(d2_prob * torch.log(d2_prob + 1e-7), dim=1).mean()
        mean_entropy = (d1_entropy + d2_entropy) / 2.0
        current_entropy_weight = 0.5 * max(0.0, 1.0 - (epoch / (epochs * 0.5)))

        # 2. Classifier Loss 
        sum_log_prob = torch.log(sum_probs + 1e-7)
        clf_loss = nll_loss_fn(sum_log_prob, labels) - (current_entropy_weight * mean_entropy)

        # 3. cVAE Loss 
        cvae_loss, _, _ = loss_function(recon_batch, data, mu, logvar)

        # 4. Prior Loss 
        z_pred = prior_net(c_pred.detach())
        prior_loss = F.mse_loss(z_pred, mu.detach())

        # 5. Joint Backprop
        loss = cvae_loss + clf_loss + prior_loss
        train_loss += loss.item()
        loss.backward()

        optimizer.step()
        classifier_optimizer.step()
        prior_optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                  f'cVAE Loss: {train_loss / len(data):.4f}\t'
                  f'Prior MSE: {prior_loss.item():.6f}')

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloader.dataset)))


def test(epoch):
    model.eval()
    classifier.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_dataloader):
            data, labels = data.to(device), labels.to(device)

            d1_logits, d2_logits = classifier(data)
            d1_gumbel = F.gumbel_softmax(d1_logits, tau=1.0, hard=True)
            d2_gumbel = F.gumbel_softmax(d2_logits, tau=1.0, hard=True)
            c_pred = torch.bmm(d1_gumbel.unsqueeze(2), d2_gumbel.unsqueeze(1)).view(-1, 100)

            recon_batch, mu, logvar = model(data, c_pred)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.detach().cpu().numpy()
            if i == 0:
                n = min(data.size(0), 5)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 1, 28, 56)[:n]])
                save_image(comparison.cpu(),
                        os.path.join(output_dir, f'reconstruction_{epoch:02}.png'), nrow=n)

    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    output_dir = "generated_images_mod5"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            c = torch.eye(100, 100).to(device)
            z_predicted = prior_net(c)
            z_sampled = z_predicted + torch.randn_like(z_predicted) * 0.1
            sample = model.decode(z_sampled, c).cpu()
            save_image(sample.view(100, 1, 28, 56),
                        os.path.join(output_dir, f'sample_{epoch:02}.png'), nrow=10)

    model_save_path = os.path.join(output_dir, "cvae_model_weights.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"====> Model weights successfully saved to {model_save_path}")

