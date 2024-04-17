import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Vanilla_NN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_hidden_layers,
    ):
        super(Vanilla_NN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers

        self.W, self.b, self.W_head, self.b_head = self._create_weights(
            input_size, hidden_size, output_size, n_hidden_layers
        )

    def _create_weights(self, input_size, hidden_size, output_size, n_hidden_layers):
        hidden_size = deepcopy(hidden_size)
        hidden_size = [hidden_size] * n_hidden_layers
        hidden_size.insert(0, input_size)
        hidden_size.append(output_size)
        W = nn.ParameterList()
        b = nn.ParameterList()
        for i in range(n_hidden_layers):
            din = hidden_size[i]
            dout = hidden_size[i + 1]
            Wi_val = torch.empty(din, dout)
            nn.init.trunc_normal_(Wi_val, mean=0, std=0.1)
            Wi_val = nn.Parameter(Wi_val, requires_grad=True)
            bi_val = torch.empty(dout)
            nn.init.trunc_normal_(bi_val, mean=0, std=0.1)
            bi_val = nn.Parameter(bi_val, requires_grad=True)
            W.append(Wi_val)
            b.append(bi_val)
        
        W_head = nn.ParameterList()
        b_head = nn.ParameterList()
        din = hidden_size[-2]
        dout = hidden_size[-1]
        W_head_val = torch.empty(din, dout)
        nn.init.trunc_normal_(W_head_val, mean=0, std=0.1)
        W_head_val = nn.Parameter(W_head_val, requires_grad=True)
        W_head.append(W_head_val)
        b_head_val = torch.empty(dout)
        nn.init.trunc_normal_(b_head_val, mean=0, std=0.1)
        b_head_val = nn.Parameter(b_head_val, requires_grad=True)
        b_head.append(b_head_val)

        return W, b, W_head, b_head
    
    def forward(self, x):
        for W, b in zip(self.W, self.b):
            x = F.relu(x @ W + b)
        out = x @ self.W_head[0] + self.b_head[0]
        return out
    
    def get_weights(self):
        return self.W, self.b, self.W_head, self.b_head
    
class MFVI_NN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_hidden_layers,
        n_heads,
        prev_weights=None,
    ):
        super(MFVI_NN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers
        self.n_heads = n_heads
        self.initial_posterior_log_var = -6.0

        m, v = self._create_weights(
            input_size, hidden_size, output_size, n_hidden_layers, n_heads, prev_weights
        )
        self.W_m, self.b_m, self.W_head_m, self.b_head_m = m[0], m[1], m[2], m[3]
        self.W_log_var, self.b_log_var, self.W_head_log_var, self.b_head_log_var = v[0], v[1], v[2], v[3]

        m, v = self._create_priors(
            input_size, hidden_size, output_size, n_hidden_layers, n_heads
        )
        self.W_prior_m, self.b_prior_m, self.W_head_prior_m, self.b_head_prior_m = m[0], m[1], m[2], m[3]
        self.W_prior_log_var, self.b_prior_log_var, self.W_head_prior_log_var, self.b_head_prior_log_var = v[0], v[1], v[2], v[3]

    def _create_weights(self, input_size, hidden_size, output_size, n_hidden_layers, n_heads, prev_weights):
        hidden_size = deepcopy(hidden_size)
        hidden_size = [hidden_size] * n_hidden_layers
        hidden_size.insert(0, input_size)
        hidden_size.append(output_size)

        W_m = nn.ParameterList()
        b_m = nn.ParameterList()
        W_log_var = nn.ParameterList()
        b_log_var = nn.ParameterList()
        for i in range(n_hidden_layers):
            din = hidden_size[i]
            dout = hidden_size[i + 1]
            if prev_weights is None:
                Wi_m_val = torch.empty(din, dout)
                nn.init.trunc_normal_(Wi_m_val, mean=0, std=0.1)
                Wi_m_val = nn.Parameter(Wi_m_val, requires_grad=True)
                bi_m_val = torch.empty(dout)
                nn.init.trunc_normal_(bi_m_val, mean=0, std=0.1)
                bi_m_val = nn.Parameter(bi_m_val, requires_grad=True)
            else:
                Wi_m_val = nn.Parameter(prev_weights[0][i], requires_grad=True)
                bi_m_val = nn.Parameter(prev_weights[1][i], requires_grad=True)

            Wi_v_val = torch.empty(din, dout)
            nn.init.constant_(Wi_v_val, self.initial_posterior_log_var)
            Wi_v_val = nn.Parameter(Wi_v_val, requires_grad=True)
            bi_v_val = torch.empty(dout)
            nn.init.constant_(bi_v_val, self.initial_posterior_log_var)
            bi_v_val = nn.Parameter(bi_v_val, requires_grad=True)

            W_m.append(Wi_m_val)
            b_m.append(bi_m_val)
            W_log_var.append(Wi_v_val)
            b_log_var.append(bi_v_val)
        
        W_head_m = nn.ParameterList()
        b_head_m = nn.ParameterList()
        W_head_log_var = nn.ParameterList()
        b_head_log_var = nn.ParameterList()
        for i in range(n_heads):
            din = hidden_size[-2]
            dout = hidden_size[-1]
            if prev_weights is None:
                W_head_m_val = torch.empty(din, dout)
                nn.init.trunc_normal_(W_head_m_val, mean=0, std=0.1)
                W_head_m_val = nn.Parameter(W_head_m_val, requires_grad=True)
                b_head_m_val = torch.empty(dout)
                nn.init.trunc_normal_(b_head_m_val, mean=0, std=0.1)
                b_head_m_val = nn.Parameter(b_head_m_val, requires_grad=True)
            else:
                W_head_m_val = nn.Parameter(prev_weights[2][0], requires_grad=True)
                b_head_m_val = nn.Parameter(prev_weights[3][0], requires_grad=True)

            W_head_v_val = torch.empty(din, dout)
            nn.init.constant_(W_head_v_val, self.initial_posterior_log_var)
            W_head_v_val = nn.Parameter(W_head_v_val, requires_grad=True)
            b_head_v_val = torch.empty(dout)
            nn.init.constant_(b_head_v_val, self.initial_posterior_log_var)
            b_head_v_val = nn.Parameter(b_head_v_val, requires_grad=True)

            W_head_m.append(W_head_m_val)
            b_head_m.append(b_head_m_val)
            W_head_log_var.append(W_head_v_val)
            b_head_log_var.append(b_head_v_val)

        return [W_m, b_m, W_head_m, b_head_m], [W_log_var, b_log_var, W_head_log_var, b_head_log_var]

    def _create_priors(self, input_size, hidden_size, output_size, n_hidden_layers, n_heads):
        hidden_size = deepcopy(hidden_size)
        hidden_size = [hidden_size] * n_hidden_layers
        hidden_size.insert(0, input_size)
        hidden_size.append(output_size)
        W_prior_m = []
        b_prior_m = []
        W_head_prior_m = []
        b_head_prior_m = []
        W_prior_log_var = []
        b_prior_log_var = []
        W_head_prior_log_var = []
        b_head_prior_log_var = []
        for i in range(n_hidden_layers):
            din = hidden_size[i]
            dout = hidden_size[i + 1]
            Wi_m_val = torch.zeros(din, dout)
            self.register_buffer("prior_W_means_" + str(i), Wi_m_val, persistent=True)
            bi_m_val = torch.zeros(dout)
            self.register_buffer("prior_b_means_" + str(i), bi_m_val, persistent=True)
            Wi_v_val = torch.zeros(din, dout)
            self.register_buffer("prior_W_log_vars_" + str(i), Wi_v_val, persistent=False)
            bi_v_val = torch.zeros(dout)
            self.register_buffer("prior_b_log_vars_" + str(i), bi_v_val, persistent=False)
            W_prior_m.append(Wi_m_val)
            b_prior_m.append(bi_m_val)
            W_prior_log_var.append(Wi_v_val)
            b_prior_log_var.append(bi_v_val)
        
        for i in range(n_heads):
            din = hidden_size[-2]
            dout = hidden_size[-1]
            Wi_m_val = torch.zeros(din, dout)
            self.register_buffer("head_prior_W_means_" + str(i), Wi_m_val, persistent=True)
            bi_m_val = torch.zeros(dout)
            self.register_buffer("head_prior_b_means_" + str(i), bi_m_val, persistent=True)
            Wi_v_val = torch.zeros(din, dout)
            self.register_buffer("head_prior_W_log_vars_" + str(i), Wi_v_val, persistent=False)
            bi_v_val = torch.zeros(dout)
            self.register_buffer("head_prior_b_log_vars_" + str(i), bi_v_val, persistent=False)
            W_head_prior_m.append(Wi_m_val)
            b_head_prior_m.append(bi_m_val)
            W_head_prior_log_var.append(Wi_v_val)
            b_head_prior_log_var.append(bi_v_val)
        
        return [W_prior_m, b_prior_m, W_head_prior_m, b_head_prior_m], [W_prior_log_var, b_prior_log_var, W_head_prior_log_var, b_head_prior_log_var]

    def forward(self, x, head):

        sampled_W, sampled_b = self._sample_weights(self.W_m, self.W_log_var, self.b_m, self.b_log_var)
        sampled_W_head, sampled_b_head = self._sample_weights(self.W_head_m, self.W_head_log_var, self.b_head_m, self.b_head_log_var)

        for W, b in zip(sampled_W, sampled_b):
            x = F.relu(x @ W + b)
        out = x @ sampled_W_head[head] + sampled_b_head[head]

        return out
    
    def _sample_weights(self, W_m, W_log_var, b_m, b_log_var):

        sampled_W, sampled_b = [], []
        for i in range(len(W_m)):
            W_eps = torch.FloatTensor(W_m[i].size()).normal_().to(W_m[i].device)
            W_std = W_log_var[i].mul(0.5).exp_()
            sampled_W.append(W_eps.mul(W_std).add_(W_m[i]))

            b_eps = torch.FloatTensor(b_m[i].size()).normal_().to(b_m[i].device)
            b_std = b_log_var[i].mul(0.5).exp_()
            sampled_b.append(b_eps.mul(b_std).add_(b_m[i]))

        return sampled_W, sampled_b
    
    def vcl_loss(self, x, y, head, task_size, n_train_samples=50):
        """
        Equation (4) from VCL paper.
        """
        kl_div = self._KL_term() / task_size
        log_likelihood = self._logpred(x, y, head, n_train_samples)
        out = kl_div - log_likelihood
        return out
    
    def _logpred(self, x, y, head, num_samples):
        log_likelihood = 0
        for _ in range(num_samples):
            logits = self.forward(x, head)
            log_likelihood += F.cross_entropy(logits, y, reduction='sum')
        log_likelihood /= num_samples
        return -log_likelihood
    
    def _KL_term(self):

        kl_div = 0
        for i in range(self.n_hidden_layers):
            mu_q, logvar_q = self.W_m[i], self.W_log_var[i]
            mu_p, logvar_p = self.W_prior_m[i], self.W_prior_log_var[i]
            var_q = torch.exp(logvar_q)
            var_p = torch.exp(logvar_p)
            kl_div = (var_q / var_p + (mu_q - mu_p)**2 / var_p - 1 + logvar_p - logvar_q).sum() * 0.5

            mu_q, logvar_q = self.b_m[i], self.b_log_var[i]
            mu_p, logvar_p = self.b_prior_m[i], self.b_prior_log_var[i]
            var_q = torch.exp(logvar_q)
            var_p = torch.exp(logvar_p)
            kl_div = (var_q / var_p + (mu_q - mu_p)**2 / var_p - 1 + logvar_p - logvar_q).sum() * 0.5
        
        for i in range(self.n_heads):
            mu_q, logvar_q = self.W_head_m[i], self.W_head_log_var[i]
            mu_p, logvar_p = self.W_head_prior_m[i], self.W_head_prior_log_var[i]
            var_q = torch.exp(logvar_q)
            var_p = torch.exp(logvar_p)
            kl_div = (var_q / var_p + (mu_q - mu_p)**2 / var_p - 1 + logvar_p - logvar_q).sum() * 0.5

            mu_q, logvar_q = self.b_head_m[i], self.b_head_log_var[i]
            mu_p, logvar_p = self.b_head_prior_m[i], self.b_head_prior_log_var[i]
            var_q = torch.exp(logvar_q)
            var_p = torch.exp(logvar_p)
            kl_div = (var_q / var_p + (mu_q - mu_p)**2 / var_p - 1 + logvar_p - logvar_q).sum() * 0.5

        return kl_div
    
    def reset_for_new_task(self, head):
        """
        Called after completion of a task, to reset state for the next task
        """
        for i in range(self.n_hidden_layers):
            self.W_prior_m[i].data.copy_(self.W_m[i].data)
            self.W_prior_log_var[i].data.copy_(self.W_log_var[i].data)
            self.b_prior_m[i].data.copy_(self.b_m[i].data)
            self.b_prior_log_var[i].data.copy_(self.b_log_var[i].data)

        # set the value of the head prior to be the current value of the posterior
        self.W_head_prior_m[head].data.copy_(self.W_head_m[head].data)
        self.W_head_prior_log_var[head].data.copy_(self.W_head_log_var[head].data)
        self.b_head_prior_m[head].data.copy_(self.b_head_m[head].data)
        self.b_head_prior_log_var[head].data.copy_(self.b_head_log_var[head].data)

    def _apply(self, fn):
        super(MFVI_NN, self)._apply(fn)
        self.W_prior_m = [fn(p) for p in self.W_prior_m]
        self.b_prior_m = [fn(p) for p in self.b_prior_m]
        self.W_head_prior_m = [fn(p) for p in self.W_head_prior_m]
        self.b_head_prior_m = [fn(p) for p in self.b_head_prior_m]
        self.W_prior_log_var = [fn(p) for p in self.W_prior_log_var]
        self.b_prior_log_var = [fn(p) for p in self.b_prior_log_var]
        self.W_head_prior_log_var = [fn(p) for p in self.W_head_prior_log_var]
        self.b_head_prior_log_var = [fn(p) for p in self.b_head_prior_log_var]
        return self
    
class FC_Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC_Generator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.generator = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        return self.generator(x)
    
class FC_Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC_Discriminator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

class Conv_Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv_Generator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 64*10*10)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=5)
        self.fc2 = nn.Linear(28*28, output_size)
        self.act = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2)

        
    def forward(self, x):
        x = self.fc1(x) # [bsz, 64*10*10]
        x = x.view(-1, 64, 10, 10) # [bsz, 64, 10, 10]
        x = self.upsample(x) # [bsz, 64, 20, 20]
        x = self.act(self.conv1(x)) # [bsz, 64, 16, 16]
        x = self.upsample(x) # [bsz, 64, 32, 32]
        x = self.act(self.conv2(x)) # [bsz, 1, 28, 28]
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3)) # [bsz, 1*28*28]
        x = self.fc2(x) # [bsz, 784]
        return x

class Conv_Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv_Discriminator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 64*10*10)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=5)
        self.fc2 = nn.Linear(28*28, output_size)
        self.act = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 64, 10, 10)
        x = self.upsample(x) # [bsz, 64, 20, 20]
        x = self.act(self.conv1(x)) # [bsz, 64, 16, 16]      
        x = self.upsample(x) # [bsz, 64, 32, 32]
        x = self.act(self.conv2(x)) # [bsz, 1, 28, 28]
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3)) # [bsz, 1*28*28]
        x = self.fc2(x) # [bsz, 1]
        x = F.sigmoid(x)
        return x

class DCGAN_MNIST_Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=28):
        super(DCGAN_MNIST_Generator, self).__init__()
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.generator(input)
        return output


class DCGAN_MNIST_Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=28):
        super(DCGAN_MNIST_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.discriminator(input)
        return output.view(-1, 1)
    
# Source: https://github.com/pytorch/examples/blob/main/dcgan/main.py
class DCGAN_CIFAR10_Generator(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64):
        super(DCGAN_CIFAR10_Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Source: https://github.com/pytorch/examples/blob/main/dcgan/main.py
class DCGAN_CIFAR10_Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(DCGAN_CIFAR10_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)