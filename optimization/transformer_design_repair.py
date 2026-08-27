import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch.amp import GradScaler
import math


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
    
    
class CustomDecoderLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=64, dropout=0.1):

        super(CustomDecoderLayer, self).__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, tgt, tgt_mask=None):

        tgt2 = F.scaled_dot_product_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class CustomTransformerDecoder(nn.Module):

    def __init__(self, d_model, num_layers, dim_feedforward=64, dropout=0.1):

        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(d_model, dim_feedforward, dropout) 
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers


    def forward(self, tgt, tgt_mask=None):

        output = tgt
        for layer in self.layers:
            output = layer(output, tgt_mask)

        return output


class Actor(nn.Module):
    """
    Takes complete design as input, outputs:
    1. Variable to modify (pointer network)
    2. New value for that variable
    3. Stop indicator
    """
    
    def __init__(self, device, params, num_variables, action_length, num_objectives):
        super(Actor, self).__init__()
        self.device = device
        self.params = params
        self.num_variables = num_variables
        self.action_range = action_length
        self.num_objectives = num_objectives
        
        self.dense_dim = 16
        self.nhead = 2
        self.num_layers = 2
        self.clip_ratio = params['clip_ratio']
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input: design variables + objectives
        self.input_dim = num_variables + num_objectives
        
        # Encode input design
        self.design_encoder = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        
        # Transformer encoder to process design
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dense_dim,
            nhead=self.nhead,
            dim_feedforward=self.dense_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Pointer network for variable selection (attention-based)
        self.pointer_query = nn.Linear(self.dense_dim, self.dense_dim)
        self.pointer_key = nn.Linear(self.dense_dim, self.dense_dim)
        
        # Value prediction head (conditional on selected variable)
        self.value_predictor = nn.Sequential(
            nn.Linear(self.dense_dim * 2, self.dense_dim),
            nn.ReLU(),
            nn.Linear(self.dense_dim, action_length)
        )
        
        # Stop decision head
        self.stop_head = nn.Sequential(
            nn.Linear(self.dense_dim, self.dense_dim // 2),
            nn.ReLU(),
            nn.Linear(self.dense_dim // 2, 2)  # [continue, stop]
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    
    
    def forward(self, design_state):
        """
        design_state: [batch, num_variables + num_objectives]
        Returns pointer logits, value logits, stop logits, and encoded features
        """
        with autocast(device_type=self.device.type, dtype=torch.float16):
            batch_size = design_state.size(0)
            
            # Encode each element of the design separately
            design_encoded = self.design_encoder(design_state.unsqueeze(-1))  # [batch, num_vars, dense_dim]
            design_encoded = self.positional_encoding(design_encoded.transpose(0, 1)).transpose(0, 1)
            
            # Transform through encoder
            encoded = self.transformer_encoder(design_encoded)  # [batch, num_vars, dense_dim]
            
            # Pointer network: which variable to modify?
            # Use mean pooling as query context
            context = encoded.mean(dim=1, keepdim=True)  # [batch, 1, dense_dim]
            query = self.pointer_query(context)  # [batch, 1, dense_dim]
            keys = self.pointer_key(encoded)  # [batch, num_vars, dense_dim]
            
            # Attention scores for variable selection
            pointer_logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)  # [batch, num_vars]
            pointer_logits = pointer_logits / math.sqrt(self.dense_dim)
            
            # Stop decision (based on context)
            stop_logits = self.stop_head(context.squeeze(1))  # [batch, 2]
            
            return pointer_logits, encoded, stop_logits
    
    
    def get_value_logits(self, encoded, var_indices):
        """
        Given encoded design and selected variable indices, predict new values
        encoded: [batch, num_vars, dense_dim]
        var_indices: [batch]
        """
        with autocast(device_type=self.device.type, dtype=torch.float16):
            batch_size = encoded.size(0)
            
            # Gather the encoded representation of selected variables
            var_indices_expanded = var_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.dense_dim)
            selected_var_encoding = torch.gather(encoded, 1, var_indices_expanded).squeeze(1)  # [batch, dense_dim]
            
            # Context from whole design
            context = encoded.mean(dim=1)  # [batch, dense_dim]
            
            # Concatenate and predict value
            combined = torch.cat([selected_var_encoding, context], dim=-1)  # [batch, dense_dim * 2]
            value_logits = self.value_predictor(combined)  # [batch, action_range]
            
            return value_logits
    
    
    def sample_action(self, design_state):
        """
        Sample variable to change, new value, and stop decision
        """
        if design_state.dim() == 1:
            design_state = design_state.unsqueeze(0)  # [1, vars+objs]
        pointer_logits, encoded, stop_logits = self.forward(design_state)

        # Sample variable to modify - use logits directly (cast to fp32)
        var_dist = torch.distributions.Categorical(logits=pointer_logits.float())
        var_indices = var_dist.sample()
        var_log_probs = var_dist.log_prob(var_indices)

        # Sample new value for selected variable
        value_logits = self.get_value_logits(encoded, var_indices)
        value_dist = torch.distributions.Categorical(logits=value_logits.float())
        value_indices = value_dist.sample()
        value_log_probs = value_dist.log_prob(value_indices)

        # Sample stop decision
        stop_dist = torch.distributions.Categorical(logits=stop_logits.float())
        stop_decisions = stop_dist.sample()
        stop_log_probs = stop_dist.log_prob(stop_decisions)

        return (var_log_probs, var_indices,
                value_log_probs, value_indices,
                stop_log_probs, stop_decisions)
    
    
    def ppo_update(self, observations, actions, logprobs, advantages):
        """
        PPO update for all three outputs
        """
        self.optimizer.zero_grad()
        
        with autocast(device_type=self.device.type, dtype=torch.float16):
            policy_losses = []
            all_kls = []
            for idx in range(len(observations)):
                observation = observations[idx]
                action = actions[idx]
                logprob = logprobs[idx]
                advantage = advantages[idx]
                # Forward pass
                pointer_logits, encoded, stop_logits = self.forward(observation)
                
                # Recompute log probs for variable selection
                var_log_probs = F.log_softmax(pointer_logits, dim=-1)
                var_new_log_probs = var_log_probs.gather(1, action[:, 0].unsqueeze(-1).long()).squeeze(-1)
                
                # Recompute log probs for value selection
                value_logits = self.get_value_logits(encoded, action[:, 0].long())
                value_log_probs = F.log_softmax(value_logits, dim=-1)
                value_new_log_probs = value_log_probs.gather(1, action[:, 1].unsqueeze(-1).long()).squeeze(-1)
                
                # Recompute log probs for stop decision
                stop_log_probs = F.log_softmax(stop_logits, dim=-1)
                stop_new_log_probs = stop_log_probs.gather(1, action[:, 2].unsqueeze(-1).long()).squeeze(-1)
                
                # Combined log prob (joint probability)
                old_log_probs = torch.sum(logprob, dim=-1)
                new_log_probs = var_new_log_probs + value_new_log_probs + stop_new_log_probs
                
                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                min_advantage = torch.where(
                    advantage > 0,
                    (1 + self.clip_ratio) * advantage,
                    (1 - self.clip_ratio) * advantage
                )
                policy_loss = torch.min(ratio * advantage, min_advantage)
                policy_losses.append(-policy_loss)
                all_kls.append(torch.mean(new_log_probs - old_log_probs))

            # Concatenate all losses and average across all timesteps
            policy_loss = torch.cat(policy_losses).mean()
        
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        kl = torch.abs(torch.stack(all_kls).mean())
        
        return policy_loss.item(), kl.item()


class Critic(nn.Module):
    """
    Transformer-based value network for design repair.
    Outputs a scalar value estimate (single-objective, no weights needed).
    Simpler than [3][6] since design_repair.py uses scalar returns directly.
    """

    def __init__(self, device, params, num_objectives, input_dim):
        super(Critic, self).__init__()
        self.device = device
        self.params = params
        self.num_objectives = num_objectives
        self.input_dim = input_dim

        self.dense_dim = 16
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        # Transformer components - mirrors Actor structure in this file [5]
        self.encoder = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim,
            num_layers=1,
            dim_feedforward=self.dense_dim,
            dropout=0.1
        )
        # Scalar output - matches existing ppo_update signature (no weights) [5]
        self.output_layer = nn.Linear(self.dense_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)


    def generate_square_subsequent_mask(self, sz):

        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask


    def forward(self, design_state):
        """
        design_state: [batch, input_dim]
        Returns: scalar value estimate [batch]
        """
        with autocast(device_type=self.device.type, dtype=torch.float16):
            x = design_state.unsqueeze(-1)               # [batch, input_dim, 1]
            x = self.encoder(x)                           # [batch, input_dim, dense_dim]
            x = self.positional_encoding(x)               # [batch, input_dim, dense_dim]
            mask = self.generate_square_subsequent_mask(x.size(1)).to(self.device)
            x = self.transformer_decoder(x, tgt_mask=mask)  # [batch, input_dim, dense_dim]
            x = self.output_layer(x)                      # [batch, input_dim, 1]
            x = x[:, -1, :].squeeze(-1)                  # [batch] - last token, scalar

        return x


    def sample_critic(self, design_states):
        """Evaluate value for batch of design states"""
        if not isinstance(design_states, torch.Tensor):
            design_states = torch.tensor(design_states, dtype=torch.float32, device=self.device)
        if design_states.device != self.device:
            design_states = design_states.to(self.device)
        # Handle extra batch dimensions safely (avoids the squeeze(1) bug from [3])
        if design_states.dim() == 3 and design_states.size(0) == 1:
            design_states = design_states.squeeze(0)
        elif design_states.dim() == 1:
            design_states = design_states.unsqueeze(0)
        return self.forward(design_states)


    def ppo_update(self, design_states, returns):
        """
        Update critic with PPO - signature unchanged from existing [5].
        No weights tensor needed since this problem uses scalar returns.
        """
        self.optimizer.zero_grad()
        value_losses = []

        with autocast(device_type=self.device.type, dtype=torch.float16):
            for idx in range(len(design_states)):
                if design_states[idx].size(0) == 0:
                    continue
                predicted_values = self.forward(design_states[idx])  # [timesteps]
                value_loss = (predicted_values - returns[idx]) ** 2   # [timesteps]
                value_losses.append(value_loss)

            if len(value_losses) == 0:
                return 0.0

            value_loss = torch.cat(value_losses).mean()

        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return value_loss.item()