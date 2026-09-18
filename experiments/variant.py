"""Parameterised copy of the synthesis PPO. cfg defaults reproduce the repo exactly."""
import os, sys, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import scipy.signal
from contextlib import nullcontext
from copy import deepcopy
from torch.amp import autocast, GradScaler
from pymoo.indicators.hv import HV
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress, RunningParetoArchive
from optimization.transformer_rand_weights_state import CustomTransformerDecoder, PositionalEncoding

DEFAULTS = dict(token_encoding='scalar',   # scalar | scaled | split | embedding
                d_model_actor=16, d_model_critic=16,
                nlayers_actor=1, nlayers_critic=1,
                entropy_coef=0.0, gamma=0.999, lam=0.9,
                amp=True, clip_ratio=0.05, learning_rate=3e-4,
                update_iterations=5, target_kl=0.003,
                dropout=0.1, eval_rollout=False, fix_position=False, fix_pe=False,
                entropy_decay=False,
                # reward target: 'scalarized' == repo behaviour, 'hv' == contribution
                # to the running Pareto front (0 for infeasible designs)
                reward_mode='scalarized', hv_scale=200.0, hv_archive='global',
                # round 4: reset the HVI archive every k epochs (1 == the round-3
                # 'batch' winner, 0/None == never, i.e. the global running archive).
                # hv_binary replaces the HVI magnitude with front membership.
                hv_reset_period=None, hv_binary=False,
                # True == repo behaviour: constraint weights pinned to 1.0 before
                # normalisation, which spends ~58% of the weight mass on dimensions
                # that are identically zero for every feasible design
                pin_constraint_weights=True)

class Encoder(nn.Module):
    """Turns [weights..., actions...] into a token sequence."""
    def __init__(self, mode, d, n_w, n_act_vals):
        super().__init__()
        self.mode, self.n_w, self.n_act_vals = mode, n_w, n_act_vals
        if mode in ('scalar', 'scaled'):
            self.shared = nn.Linear(1, d)
        else:
            self.w_enc = nn.Linear(1, d)
            if mode == 'embedding':
                self.a_emb = nn.Embedding(n_act_vals + 1, d)   # +1 = pad
            else:
                self.a_enc = nn.Linear(1, d)

    def forward(self, x):
        w, a = x[:, :self.n_w], x[:, self.n_w:]
        if self.mode == 'scalar':
            return self.shared(x.unsqueeze(-1))
        if self.mode == 'scaled':
            xs = torch.cat([w, a / (self.n_act_vals - 1)], dim=1)
            return self.shared(xs.unsqueeze(-1))
        wt = self.w_enc(w.unsqueeze(-1))
        if a.shape[1] == 0:
            return wt
        if self.mode == 'embedding':
            at = self.a_emb(a.long().clamp(0, self.n_act_vals))
        else:
            at = self.a_enc(a.unsqueeze(-1))
        return torch.cat([wt, at], dim=1)

def _mask(sz, dev):
    m = torch.triu(torch.ones(sz, sz, device=dev), diagonal=1)
    return m.masked_fill(m == 1, float('-inf'))

class Actor(nn.Module):
    def __init__(self, device, cfg, action_length, n_w):
        super().__init__()
        self.device, self.cfg, self.action_length = device, cfg, action_length
        d = cfg['d_model_actor']
        self.amp = cfg['amp']
        self.clip_ratio = cfg['clip_ratio']
        self.entropy_coef = cfg['entropy_coef']
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.amp)
        self.encoder = Encoder(cfg['token_encoding'], d, n_w, action_length)
        self.positional_encoding = PositionalEncoding(d, dropout=cfg['dropout'])
        self.transformer_decoder = CustomTransformerDecoder(d_model=d, num_layers=cfg['nlayers_actor'],
                                                            dim_feedforward=d, dropout=cfg['dropout'])
        self.output_layer = nn.Linear(d, action_length)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.skipped = 0; self.steps = 0

    def _ctx(self):
        return autocast(device_type=self.device.type, dtype=torch.float16) if self.amp else nullcontext()

    def forward(self, x):
        with self._ctx():
            h = self.encoder(x)
            h = (self.positional_encoding(h.transpose(0, 1)).transpose(0, 1)
                 if self.cfg['fix_pe'] else self.positional_encoding(h))
            h = self.transformer_decoder(h, tgt_mask=_mask(h.size(1), self.device))
            return F.softmax(self.output_layer(h), dim=-1)

    def sample_action(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device)
        out = self.forward(obs)[:, -1, :]
        logp = torch.log(out + 1e-10)
        dist = torch.distributions.Categorical(logits=logp.float())
        a = dist.sample()
        return dist.log_prob(a), a, dist.entropy().mean().item()

    def ppo_update(self, obs, act, logp_old, adv, pos=None):
        self.optimizer.zero_grad()
        with self._ctx():
            out = self.forward(obs)
            probs = out[torch.arange(out.size(0), device=out.device), pos, :] if pos is not None else out[:, -1, :]
            logp_all = torch.log(probs + 1e-10)
            new_logp = torch.sum(F.one_hot(act.long(), self.action_length) * logp_all, dim=-1)
            ratio = torch.exp(new_logp - logp_old)
            min_adv = torch.where(adv > 0, (1 + self.clip_ratio) * adv, (1 - self.clip_ratio) * adv)
            loss = -torch.mean(torch.min(ratio * adv, min_adv))
            if self.entropy_coef:
                ent = -(probs.float() * logp_all.float()).sum(-1).mean()
                loss = loss - self.entropy_coef * ent
        prev = self.scaler.get_scale() if self.amp else None
        self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
        self.steps += 1
        if self.amp and self.scaler.get_scale() < prev: self.skipped += 1
        self.scheduler.step()
        kl = torch.abs(torch.mean(new_logp - logp_old))
        return loss.item(), kl.item()

class Critic(nn.Module):
    def __init__(self, device, cfg, num_objectives, input_dim, action_length, n_w):
        super().__init__()
        self.device, self.cfg, self.input_dim = device, cfg, input_dim
        d = cfg['d_model_critic']; self.amp = cfg['amp']
        self.pad = action_length          # pad id for the embedding variant
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.amp)
        self.encoder = Encoder(cfg['token_encoding'], d, n_w, action_length)
        self.positional_encoding = PositionalEncoding(d, dropout=cfg['dropout'])
        self.transformer_decoder = CustomTransformerDecoder(d_model=d, num_layers=cfg['nlayers_critic'],
                                                            dim_feedforward=d, dropout=cfg['dropout'])
        self.output_layer = nn.Linear(d, num_objectives)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

    def _ctx(self):
        return autocast(device_type=self.device.type, dtype=torch.float16) if self.amp else nullcontext()

    def forward(self, x):
        with self._ctx():
            h = self.encoder(x)
            h = (self.positional_encoding(h.transpose(0, 1)).transpose(0, 1)
                 if self.cfg['fix_pe'] else self.positional_encoding(h))
            h = self.transformer_decoder(h, tgt_mask=_mask(h.size(1), self.device))
            return self.output_layer(h)[:, -1, :]

    def sample_critic(self, observation, n_w):
        padval = self.pad if self.cfg['token_encoding'] == 'embedding' else 0
        rows = []
        for obs in observation:
            o = list(obs) + [padval] * (self.input_dim - len(obs))
            rows.append(o[:self.input_dim])
        return self(torch.tensor(rows, dtype=torch.float32, device=self.device))

    def ppo_update(self, obs, ret, weights):
        self.optimizer.zero_grad()
        with self._ctx():
            v = self(obs)
            pred = torch.sum(-v * weights, dim=-1)
            loss = torch.mean((pred - ret) ** 2)
        self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
        self.scheduler.step()
        return loss.item()

def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def run_ppo(cfg, max_values, seed, epochs, batch, log_every=1):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob = TrussProblem()
    nobj = prob.num_objectives; omm = prob.objective_min_max
    mo = max_values[:nobj]; min_mask = np.array([o.lower() == 'min' for o in omm])
    nact = prob.n_elements; arange = len(prob.area_map); ntot = len(max_values)
    hv_ind = HV(ref_point=np.ones(nobj))

    def to_hv(o):
        n = np.array(o, dtype=float) / mo
        return np.clip(np.where(min_mask[:nobj], n, 1.0 - n), 0.0, 1.0)

    archive = RunningParetoArchive(hv_ind)

    actor = Actor(dev, cfg, arange, ntot).to(dev)
    critic = Critic(dev, cfg, ntot, nact + ntot, arange, ntot).to(dev)

    all_des, all_obj, all_con, all_cv = [], [], [], []
    ent_hist, ev_hist, kl_hist = [], [], []
    t0 = time.perf_counter()

    for ep in range(epochs):
        period = cfg['hv_reset_period']
        if period is None:
            period = 1 if cfg['hv_archive'] == 'batch' else 0
        if period and ep % period == 0:
            archive = RunningParetoArchive(hv_ind)
        if cfg['entropy_decay']:
            actor.entropy_coef = cfg['entropy_coef'] * max(0.0, 1.0 - ep / epochs)
        rewards = [[] for _ in range(batch)]; actions = [[] for _ in range(batch)]
        logps = [[] for _ in range(batch)]; designs = [[] for _ in range(batch)]
        wn = np.random.rand(batch, ntot)
        if cfg['pin_constraint_weights']: wn[:, -2:] = 1.0
        weights = wn / wn.sum(axis=1, keepdims=True)
        obs = [weights[i].tolist() for i in range(batch)]

        ents = []
        if cfg['eval_rollout']: actor.eval(); critic.eval()
        with torch.no_grad():
            for _ in range(nact):
                lp, a, e = actor.sample_action(obs); ents.append(e)
                for i, act in enumerate(a.tolist()):
                    actions[i].append(act); logps[i].append(lp[i].item())
                    obs[i].append(act); rewards[i].append(0.); designs[i].append(int(act))
        ent_hist.append(float(np.mean(ents)))

        if cfg['eval_rollout']: actor.train(); critic.train()
        for i, d in enumerate(designs):
            ov, con, cv = prob.evaluate(d)
            all_des.append(d); all_obj.append(ov); all_con.append(con); all_cv.append(cv)
            if cfg['reward_mode'] == 'hv':
                # contribution to the running front; infeasible designs add nothing
                if con:
                    r = 0.0
                else:
                    joined, dhv = archive.offer(to_hv(ov))
                    r = (1.0 if joined else 0.0) if cfg['hv_binary'] else dhv * cfg['hv_scale']
                rewards[i][-1] += float(r)
            else:
                arr = np.array(ov + cv, float)
                if con:
                    rv = np.where(min_mask, -np.ones(ntot), 0.0)
                    rv[-2:] = np.array(cv); rv[-2:][min_mask[-2:]] = -rv[-2:][min_mask[-2:]]
                else:
                    n = arr / np.array(max_values, float)
                    rv = np.where(min_mask, -n, n)
                rewards[i][-1] += float(np.dot(weights[i], rv))

        cvals = []
        with torch.no_grad():
            for k in range(nact):
                co = [obs[i][:k + 1 + ntot] for i in range(batch)]
                cv_ = np.array(critic.sample_critic(co, ntot).float().tolist())
                cvals.append(np.sum(np.multiply(weights, cv_), axis=1))
        values = [[cvals[k][i] for k in range(nact)] for i in range(batch)]
        for i in range(batch): values[i].append(values[i][-1])

        advs, rets = [], []
        for i in range(batch):
            r = np.array(rewards[i]); v = np.array(values[i])
            deltas = r + cfg['gamma'] * v[1:] - v[:-1]
            advs.append(discounted_cumulative_sums(deltas, cfg['gamma'] * cfg['lam']))
            rets.append(np.array(discounted_cumulative_sums(r, cfg['gamma'] * cfg['lam']), np.float32))

        # explained variance of the critic against the returns it is trained on
        vflat = np.concatenate([np.array(values[i][:nact]) for i in range(batch)])
        rflat = np.concatenate(rets)
        ev_hist.append(float(1 - np.var(rflat - vflat) / (np.var(rflat) + 1e-12)))

        af = np.concatenate(advs); am, asd = af.mean(), af.std()
        advs = [(a - am) / (asd + 1e-8) for a in advs]

        O, A, L, ADV, R, W, POS = [], [], [], [], [], [], []
        for i in range(batch):
            for s in range(nact):
                frag = obs[i][:s + 1 + ntot]
                pad = actor.action_length if cfg['token_encoding'] == 'embedding' else 0
                O.append(frag + [pad] * (nact + ntot - len(frag))); POS.append(ntot + s - 1)
                A.append(actions[i][s]); L.append(logps[i][s])
                ADV.append(advs[i][s]); R.append(rets[i][s]); W.append(weights[i])
        O = torch.tensor(O, dtype=torch.float32, device=dev)
        A = torch.tensor(A, dtype=torch.float32, device=dev)
        L = torch.tensor(L, dtype=torch.float32, device=dev)
        ADV = torch.tensor(ADV, dtype=torch.float32, device=dev)
        R = torch.tensor(R, dtype=torch.float32, device=dev)
        W = torch.tensor(np.array(W), dtype=torch.float32, device=dev)
        POS = torch.tensor(POS, device=dev) if cfg['fix_position'] else None

        for _ in range(cfg['update_iterations']):
            _, kl = actor.ppo_update(O, A, L, ADV, POS)
            if kl > cfg['target_kl']: break
        kl_hist.append(kl)
        for _ in range(cfg['update_iterations']):
            critic.ppo_update(O, R, W)

    all_des = np.array(all_des); all_obj = np.array(all_obj); all_con = np.array(all_con)
    rv = mo * np.ones(nobj); rv[~min_mask[:nobj]] = 0.0
    all_obj[all_con] = rv
    norm = all_obj / mo; hv_obj = deepcopy(norm)
    for i in range(nobj):
        if omm[i].lower() == 'max': hv_obj[:, i] = 1 - norm[:, i]
    _, _, hvs = track_pareto_progress(hv_obj, all_des, all_con, hv_ind)

    return dict(hv=float(hvs[-1]), entropy=ent_hist, ev=ev_hist, kl=kl_hist,
                valid_frac=float((~all_con).mean()), wall=time.perf_counter() - t0,
                skipped=actor.skipped, steps=actor.steps,
                actor_params=sum(p.numel() for p in actor.parameters()),
                critic_params=sum(p.numel() for p in critic.parameters()),
                actor=actor, weights_dim=ntot, problem=prob, max_values=max_values)
