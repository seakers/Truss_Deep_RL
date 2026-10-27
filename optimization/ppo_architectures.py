import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DENSE_DIM, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT = 16, 2, 64, 0.1
WARMUP_STEPS = 200


def make_scheduler(optimizer):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=max(1, WARMUP_STEPS))
    decay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    if WARMUP_STEPS <= 0:
        return decay
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[WARMUP_STEPS])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        tgt2 = F.scaled_dot_product_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, num_layers, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(d_model, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, tgt, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, tgt_mask)
        return tgt


class Actor(nn.Module):
    def __init__(self, device, params, gene_space, unique_des_space, num_objectives):
        super().__init__()
        self.device = device
        self.gene_space = gene_space
        self.num_objectives = num_objectives
        self.clip_ratio = params["clip_ratio"]
        self.dense_dim = params.get("dense_dim", DENSE_DIM)
        self.num_layers = params.get("num_layers", NUM_LAYERS)

        self.encoder = nn.Linear(1, self.dense_dim)
        self.context_dim = params.get("context_dim", None)
        self.context_proj = (nn.Linear(self.context_dim, self.dense_dim)
                            if self.context_dim else None)
        self.context_norm = nn.LayerNorm(self.dense_dim) if self.context_dim else None
        self._context = None   # set via set_context(); None => no conditioning
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim, num_layers=self.num_layers,
            dim_feedforward=params.get("dim_feedforward", DIM_FEEDFORWARD),
            dropout=params.get("dropout", DROPOUT))

        for var in unique_des_space:
            assert var["type"] == "discrete"
        self.unique_names = [v["name"] for v in unique_des_space]
        assert len(set(self.unique_names)) == len(self.unique_names)
        self.name_to_head = {n: k for k, n in enumerate(self.unique_names)}
        self.heads = nn.ModuleList(
            [nn.Linear(self.dense_dim, len(v["range"])) for v in unique_des_space])
        self.head_ranges = [list(v["range"]) for v in unique_des_space]
        self.gene_to_head = [self.name_to_head[v["name"]] for v in gene_space]
        self.num_genes = len(gene_space)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["learning_rate"])
        self.scheduler = make_scheduler(self.optimizer)

    def _tokens(self, weights):
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return w.mean().repeat(self.num_genes).unsqueeze(0).unsqueeze(-1)

    def set_context(self, context):
        """context: 1D array of length context_dim, or None to disable conditioning."""
        if context is None or self.context_proj is None:
            self._context = None
        else:
            self._context = torch.as_tensor(context, dtype=torch.float32,
                                            device=self.device)

    def _add_context(self, x):
        """x: [batch, seq, dense_dim]. Adds projected context to every token (broadcast)."""
        if self._context is None or self.context_proj is None:
            return x
        c = self.context_proj(self._context)          # [dense_dim]
        return self.context_norm(x + c.view(1, 1, -1))

    def forward(self, weights):
        x = self.encoder(self._tokens(weights))
        x = self._add_context(x)                       # NEW
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        return self.transformer_decoder(x).squeeze(0)

    def sample_action(self, weights, eval_function):
        feats = self.forward(weights)
        genes, log_prob = [], torch.zeros((), device=self.device)
        for i, h in enumerate(self.gene_to_head):
            options = self.head_ranges[h]
            logits = self.heads[h](feats[i]).float()
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
            log_prob = log_prob + dist.log_prob(idx)
            genes.append(options[int(idx)])
        return np.array(genes, dtype=float), log_prob

    def ppo_update(self, weights_batch, genes_batch, masks_batch, old_logprobs, advantages):
        self.optimizer.zero_grad()
        new_logprobs = []
        for weights, genes in zip(weights_batch, genes_batch):
            feats = self.forward(weights)
            lp = torch.zeros((), device=self.device)
            for i, h in enumerate(self.gene_to_head):
                options = self.head_ranges[h]
                dist = torch.distributions.Categorical(logits=self.heads[h](feats[i]).float())
                target = torch.tensor(int(np.argmin(np.abs(np.array(options) - genes[i]))),
                                      device=self.device)
                lp = lp + dist.log_prob(target)
            new_logprobs.append(lp)
        new_logprobs = torch.stack(new_logprobs)
        old_logprobs = torch.as_tensor(old_logprobs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped = torch.where(advantages > 0, (1 + self.clip_ratio) * advantages,
                              (1 - self.clip_ratio) * advantages)
        policy_loss = -torch.mean(torch.min(ratio * advantages, clipped))
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step(); self.scheduler.step()
        kl = torch.mean(old_logprobs - new_logprobs)
        return policy_loss.item(), kl.item()


class Critic(nn.Module):
    """Transformer over the emitted design -> [obj..., cval...]. MSE training.
    Scalarization with weights happens outside the model."""

    def __init__(self, device, params, num_objectives, num_constraints):
        super().__init__()
        self.device = device
        self.out_dim = num_objectives + num_constraints        # 3 + 2 = 5 [2]
        self.dense_dim = params.get("dense_dim", DENSE_DIM)
        self.num_layers = params.get("num_layers", NUM_LAYERS)

        self.encoder = nn.Linear(1, self.dense_dim)
        self.context_dim = params.get("context_dim", None)
        self.context_proj = (nn.Linear(self.context_dim, self.dense_dim)
                            if self.context_dim else None)
        self.context_norm = nn.LayerNorm(self.dense_dim) if self.context_dim else None
        self._context = None   # set via set_context(); None => no conditioning
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim, num_layers=self.num_layers,
            dim_feedforward=params.get("dim_feedforward", DIM_FEEDFORWARD),
            dropout=params.get("dropout", DROPOUT))
        self.output_layer = nn.Linear(self.dense_dim, self.out_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["learning_rate"])
        self.scheduler = make_scheduler(self.optimizer)

    def set_context(self, context):
        """context: 1D array of length context_dim, or None to disable conditioning."""
        if context is None or self.context_proj is None:
            self._context = None
        else:
            self._context = torch.as_tensor(context, dtype=torch.float32,
                                            device=self.device)

    def _add_context(self, x):
        """x: [batch, seq, dense_dim]. Adds projected context to every token (broadcast)."""
        if self._context is None or self.context_proj is None:
            return x
        c = self.context_proj(self._context)          # [dense_dim]
        return self.context_norm(x + c.view(1, 1, -1))

    def forward(self, design):
        x = torch.as_tensor(design, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.encoder(x.unsqueeze(-1))                      # [batch, num_genes, dense]
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_decoder(x)
        return self.output_layer(x)[:, -1, :]                  # [batch, out_dim]

    def value(self, design):
        with torch.no_grad():
            return self.forward(design).squeeze(0)             # [out_dim]

    def ppo_update(self, designs_batch, targets_batch):
        """MSE against realized [obj..., cval...]."""
        self.optimizer.zero_grad()
        preds = self.forward(np.array(designs_batch))
        targets = torch.as_tensor(np.array(targets_batch), dtype=torch.float32,
                                  device=self.device)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step(); self.scheduler.step()
        return loss.item()


class RepairActor(nn.Module):
    def __init__(self, device, params, gene_space, unique_des_space, num_objectives):
        super().__init__()
        self.device = device
        self.gene_space = gene_space
        self.num_genes = len(gene_space)
        self.clip_ratio = params["clip_ratio"]
        self.dense_dim = params.get("dense_dim", DENSE_DIM)
        self.num_layers = params.get("num_layers", NUM_LAYERS)

        self.encoder = nn.Linear(1, self.dense_dim)
        self.context_dim = params.get("context_dim", None)
        self.context_proj = (nn.Linear(self.context_dim, self.dense_dim)
                            if self.context_dim else None)
        self.context_norm = nn.LayerNorm(self.dense_dim) if self.context_dim else None
        self._context = None   # set via set_context(); None => no conditioning
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim, num_layers=self.num_layers,
            dim_feedforward=params.get("dim_feedforward", DIM_FEEDFORWARD),
            dropout=params.get("dropout", DROPOUT))

        self.pointer_head = nn.Linear(self.dense_dim, 1)
        self.stop_head = nn.Linear(self.dense_dim, 2)

        self.unique_names = [v["name"] for v in unique_des_space]
        assert len(set(self.unique_names)) == len(self.unique_names)
        self.name_to_head = {n: k for k, n in enumerate(self.unique_names)}
        self.value_heads = nn.ModuleList(
            [nn.Linear(self.dense_dim, len(v["range"])) for v in unique_des_space])  # 'Sel' -> 2 logits
        self.head_ranges = [list(v["range"]) for v in unique_des_space]
        self.gene_to_head = [self.name_to_head[v["name"]] for v in gene_space]

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["learning_rate"])
        self.scheduler = make_scheduler(self.optimizer)

    def _encode(self, design):
        x = torch.as_tensor(design, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.encoder(x.unsqueeze(-1))
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        return self.transformer_decoder(x)

    def set_context(self, context):
        """context: 1D array of length context_dim, or None to disable conditioning."""
        if context is None or self.context_proj is None:
            self._context = None
        else:
            self._context = torch.as_tensor(context, dtype=torch.float32,
                                            device=self.device)

    def _add_context(self, x):
        """x: [batch, seq, dense_dim]. Adds projected context to every token (broadcast)."""
        if self._context is None or self.context_proj is None:
            return x
        c = self.context_proj(self._context)          # [dense_dim]
        return self.context_norm(x + c.view(1, 1, -1))

    def sample_action(self, design):
        """Pick a gene, a new value from that gene's own head, and a stop decision."""
        f = self._encode(design).squeeze(0)                 # [num_genes, dense]

        # 1) pointer: which gene to modify
        ptr_logits = self.pointer_head(f).squeeze(-1).float()
        var_dist = torch.distributions.Categorical(logits=ptr_logits)
        vi = int(var_dist.sample())
        var_log_prob = var_dist.log_prob(torch.tensor(vi, device=self.device))

        # 2) value: routed to the head for THIS gene's type.
        #    'Sel' -> 2 logits, 'Area' -> len(area_map) logits.
        h = self.gene_to_head[vi]
        options = self.head_ranges[h]
        val_logits = self.value_heads[h](f[vi]).float()
        val_dist = torch.distributions.Categorical(logits=val_logits)
        val_choice = int(val_dist.sample())
        value_log_prob = val_dist.log_prob(torch.tensor(val_choice, device=self.device))
        new_value = options[val_choice]

        # 3) stop decision
        stop_logits = self.stop_head(f.mean(dim=0)).float()
        stop_dist = torch.distributions.Categorical(logits=stop_logits)
        stop_decision = stop_dist.sample()
        stop_log_prob = stop_dist.log_prob(stop_decision)

        info = {'var_idx': vi, 'val_choice': val_choice, 'stop': int(stop_decision)}
        total_log_prob = var_log_prob + value_log_prob + stop_log_prob
        return vi, new_value, int(stop_decision), total_log_prob, info

    def ppo_update(self, designs, actions_info, old_logprobs, advantages, micro_bs=8):
        old_logprobs = torch.as_tensor(old_logprobs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        n = len(designs)
        self.optimizer.zero_grad()
        kl_new_logprobs = []

        for start in range(0, n, micro_bs):
            sl = slice(start, start + micro_bs)
            new_lp = []
            for design, info in zip(designs[sl], actions_info[sl]):
                f = self._encode(design).squeeze(0)
                ptr_logits = self.pointer_head(f).squeeze(-1).float()
                var_dist = torch.distributions.Categorical(logits=ptr_logits)
                lp = var_dist.log_prob(torch.tensor(info['var_idx'], device=self.device))

                vi = info['var_idx']
                h = self.gene_to_head[vi]
                val_dist = torch.distributions.Categorical(
                    logits=self.value_heads[h](f[vi]).float())
                lp = lp + val_dist.log_prob(torch.tensor(info['val_choice'], device=self.device))

                stop_dist = torch.distributions.Categorical(
                    logits=self.stop_head(f.mean(dim=0)).float())
                lp = lp + stop_dist.log_prob(torch.tensor(info['stop'], device=self.device))
                new_lp.append(lp)

            new_lp = torch.stack(new_lp)
            ratio = torch.exp(new_lp - old_logprobs[sl])
            adv_chunk = advantages[sl]
            clipped = torch.where(adv_chunk > 0, (1 + self.clip_ratio) * adv_chunk,
                                  (1 - self.clip_ratio) * adv_chunk)
            loss = -torch.sum(torch.min(ratio * adv_chunk, clipped)) / n
            loss.backward()
            kl_new_logprobs.append(new_lp.detach())

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step(); self.scheduler.step()
        new_full = torch.cat(kl_new_logprobs)
        kl = torch.mean(old_logprobs - new_full)
        policy_loss = -torch.mean(torch.min(
            torch.exp(new_full - old_logprobs) * advantages,
            torch.where(advantages > 0, (1 + self.clip_ratio) * advantages,
                        (1 - self.clip_ratio) * advantages)))
        return policy_loss.item(), kl.item()


class RepairCritic(nn.Module):
    """Scalar-value transformer critic regressing discounted returns."""

    def __init__(self, device, params, num_genes):
        super().__init__()
        self.device = device
        self.num_genes = num_genes
        self.dense_dim = params.get("dense_dim", DENSE_DIM)
        self.num_layers = params.get("num_layers", NUM_LAYERS)

        self.encoder = nn.Linear(1, self.dense_dim)
        self.context_dim = params.get("context_dim", None)
        self.context_proj = (nn.Linear(self.context_dim, self.dense_dim)
                            if self.context_dim else None)
        self.context_norm = nn.LayerNorm(self.dense_dim) if self.context_dim else None
        self._context = None   # set via set_context(); None => no conditioning
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim, num_layers=self.num_layers,
            dim_feedforward=params.get("dim_feedforward", DIM_FEEDFORWARD),
            dropout=params.get("dropout", DROPOUT))
        self.output_layer = nn.Linear(self.dense_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["learning_rate"])
        self.scheduler = make_scheduler(self.optimizer)

    def set_context(self, context):
        """context: 1D array of length context_dim, or None to disable conditioning."""
        if context is None or self.context_proj is None:
            self._context = None
        else:
            self._context = torch.as_tensor(context, dtype=torch.float32,
                                            device=self.device)

    def _add_context(self, x):
        """x: [batch, seq, dense_dim]. Adds projected context to every token (broadcast)."""
        if self._context is None or self.context_proj is None:
            return x
        c = self.context_proj(self._context)          # [dense_dim]
        return self.context_norm(x + c.view(1, 1, -1))

    def forward(self, design):
        x = torch.as_tensor(design, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.encoder(x.unsqueeze(-1))
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_decoder(x)
        return self.output_layer(x)[:, -1, 0]                  # [batch] scalar

    def value(self, design):
        with torch.no_grad():
            return float(self.forward(design)[0])

    def ppo_update(self, designs, returns):
        self.optimizer.zero_grad()
        preds = self.forward(np.array(designs))
        target = torch.as_tensor(np.array(returns), dtype=torch.float32, device=self.device)
        loss = F.mse_loss(preds, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step(); self.scheduler.step()
        return loss.item()