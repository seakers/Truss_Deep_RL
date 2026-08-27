import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
import math
from scipy.stats import norm



class CustomDecoderLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=64, dropout=0.1):

        super(CustomDecoderLayer, self).__init__()

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, tgt, tgt_mask=None):
        # Self-attention layer
        tgt2 = F.scaled_dot_product_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Feed-forward layer
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
    


class CustomTransformerDecoder(nn.Module):

    def __init__(self, d_model, num_layers, dim_feedforward=64, dropout=0.1):

        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([CustomDecoderLayer(d_model, dim_feedforward, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers


    def forward(self, tgt, tgt_mask=None):

        output = tgt
        for layer in self.layers:
            output = layer(output, tgt_mask)

        return output
    


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)
    


class Actor(nn.Module):

    def __init__(self, device, params, action_length):

        super(Actor, self).__init__()
        self.device = device
        self.params = params
        self.action_length = action_length

        self.nhead = 2
        self.dense_dim = 16
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_ratio = params['clip_ratio']
        
        self.encoder = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim,
            num_layers=1,
            dim_feedforward=self.dense_dim,
            dropout=0.1
        )
        self.output_layer = nn.Linear(self.dense_dim, self.action_length)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)


    def generate_square_subsequent_mask(self, sz):

        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask
    

    def forward(self, x):

        with autocast(device_type=self.device.type, dtype=torch.float16):
            x = x.unsqueeze(-1)
            x = self.encoder(x)
            x = self.positional_encoding(x)
            mask = self.generate_square_subsequent_mask(x.size(1)).to(self.device) # x.size(1) is the sequence length
            x = self.transformer_decoder(x, tgt_mask=mask)
            x = self.output_layer(x)
            x = F.softmax(x, dim=-1)

        return x
    

    def sample_action(self, observation):

        if len(observation[0]) == 0:
            observation = [[0] for x in range(len(observation))]
        input_observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        output = self.forward(input_observation)
        output_last = output[:, -1, :]

        log_probs = torch.log(output_last + 1e-10)
        dist = torch.distributions.categorical.Categorical(logits=log_probs)
        samples = dist.sample()
        action_ids = samples.squeeze()
        action_probs = log_probs.gather(1, action_ids.unsqueeze(-1)).squeeze()

        return action_probs, action_ids
    

    def ppo_update(self, observation, action, logprob, advantage):

        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            num_vars = observation.size(1)
            new_log_probs = torch.zeros_like(action, dtype=torch.float32).to(self.device)

            pred_probs = self.forward(observation)[:,-1,:]
            new_log_probs = torch.log(pred_probs + 1e-10)
            new_log_probs = torch.sum(
                    torch.mul(F.one_hot(action.long(), num_classes=self.action_length), new_log_probs), 
                    dim=-1
                )

            ratio = torch.exp(new_log_probs - logprob)
            min_advantage = torch.where(
                advantage > 0,
                (1 + self.clip_ratio) * advantage,
                (1 - self.clip_ratio) * advantage
            )
            policy_loss = -torch.mean(torch.min(ratio * torch.t(advantage), min_advantage))

        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        kl = torch.abs(torch.mean(new_log_probs - logprob))

        return policy_loss.item(), kl.item()
    


class Critic(nn.Module):

    def __init__(self, device, params, num_objectives, input_dim):

        super(Critic, self).__init__()
        self.device = device
        self.params = params

        self.nhead = 2
        self.dense_dim = 16
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_objectives = num_objectives
        self.input_dim = input_dim

        self.input_layer = nn.Linear(self.input_dim, self.dense_dim)
        self.hidden_layer = nn.Linear(self.dense_dim, self.dense_dim)
        self.output_layer = nn.Linear(self.dense_dim, self.num_objectives)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    

    def forward(self, x):

        with autocast(device_type=self.device.type, dtype=torch.float16):
            x = self.input_layer(x)
            x = self.hidden_layer(x)
            x = self.output_layer(x)

        return x
    

    def sample_critic(self, observation):

        input_observations = []
        for obs in observation:
            input_obs = []
            input_obs.extend(obs)
            while(len(input_obs)) < self.input_dim:
                input_obs.extend([0])
            input_observations.append(input_obs)
        input_observations = torch.tensor(input_observations, dtype=torch.float32, device=self.device)
        output = self(input_observations)  # shape: (num components, 5)

        return output
    

    def ppo_update(self, observation, return_tensor, weights_tensor):

        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            new_pred_values = self(observation)
            new_pred_reward = torch.sum(-new_pred_values * weights_tensor, dim=-1)

            value_loss = torch.mean((new_pred_reward - return_tensor) ** 2)

        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return value_loss.item()