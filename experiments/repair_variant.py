"""Repair PPO with a switchable reward target. 'domination' == current repo behaviour."""
import sys, os, time, warnings, numpy as np, torch
from copy import deepcopy
from pymoo.indicators.hv import HV
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress
from optimization.transformer_design_repair import Actor, Critic
from optimization.design_repair import (calculate_gae, generate_initial_designs, sample_next_batch)

class Archive:
    """Running Pareto front in normalised minimisation space."""
    def __init__(self, hv_ind):
        self.pts = None; self.hv_ind = hv_ind; self.hv = 0.0
    def offer(self, p):
        """Return (joined_front, delta_hv) and update."""
        if self.pts is None:
            self.pts = p[None, :]; new = self.hv_ind(self.pts)
            d = new - self.hv; self.hv = new; return True, d
        if np.any(np.all(self.pts <= p, axis=1)):
            return False, 0.0
        keep = ~np.all(p <= self.pts, axis=1)
        self.pts = np.vstack([self.pts[keep], p])
        new = self.hv_ind(self.pts); d = new - self.hv; self.hv = new
        return True, d

def step_reward(cfg, parent_obj, child_obj, infeasible, min_mask, archive, to_hv):
    pen = cfg.get('step_penalty', 0.01)
    if infeasible:
        return -1.0 - pen
    mode = cfg['reward']
    if mode == 'domination':
        pa, ca = np.array(parent_obj), np.array(child_obj)
        better = np.where(min_mask[:len(pa)], ca < pa, ca > pa)
        worse  = np.where(min_mask[:len(pa)], ca > pa, ca < pa)
        if np.any(better) and not np.any(worse): return 1.0 - pen
        if np.any(worse) and not np.any(better): return -0.5 - pen
        return -pen
    joined, dhv = archive.offer(to_hv(child_obj))
    if mode == 'front':
        return (1.0 - pen) if joined else (-0.5 - pen)
    return dhv * cfg.get('hv_scale', 200.0) - pen        # mode == 'hv'

def run_repair(cfg, max_values, seed, epochs, batch):
    cfg = {'reward':'domination','gamma':0.999,'lam':0.9,'learning_rate':3e-4,
           'clip_ratio':0.05,'update_iterations':5,'target_kl':0.003,
           # round 4: 'full' == repo behaviour, which rolls out on a 72-token design
           # but runs ppo_update on a 77-token design+objective+cv state. 'design'
           # makes the update state identical to the one the policy acted in, which
           # is also the state intelligent_mutation_ga feeds it.
           'obs_mode':'full',
           # reset the HVI archive every k epochs; 0/None == the global running archive
           'hv_reset_period':0, **cfg}
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob = TrussProblem()
    nobj = prob.num_objectives; omm = prob.objective_min_max
    nact = prob.n_elements; arange = len(prob.area_map); ntot = len(max_values)
    mo = max_values[:nobj]; min_mask = np.array([o.lower()=='min' for o in omm])
    hv_ind = HV(ref_point=np.ones(nobj))
    def to_hv(o):
        n = np.array(o, float)/mo; h = n.copy()
        for i in range(nobj):
            if omm[i].lower()=='max': h[i] = 1-n[i]
        return h

    actor  = Actor(device=dev, params=cfg, num_variables=nact,
                   action_length=arange, num_objectives=ntot).to(dev)
    critic = Critic(device=dev, params=cfg, num_objectives=ntot,
                    input_dim=nact+ntot).to(dev)
    actor(torch.zeros((1,nact+ntot),device=dev)); critic(torch.zeros((1,nact+ntot),device=dev))

    archive = Archive(hv_ind)
    all_des, all_obj, all_con = [], [], []
    NFE = 0; budget = epochs*batch; t0 = time.perf_counter(); rew_hist=[]

    epoch = 0
    while NFE < budget:
        if cfg['hv_reset_period'] and epoch % cfg['hv_reset_period'] == 0:
            archive = Archive(hv_ind)
        epoch += 1
        if NFE == 0:
            designs, objs, all_des, all_obj, all_con, NFE = generate_initial_designs(
                all_des, all_obj, all_con, prob, batch, nact, arange, NFE)
        else:
            designs, objs, all_des, all_obj, all_con, NFE = sample_next_batch(
                all_des, all_obj, all_con, prob, batch, nact, arange, NFE)

        b_obs=[[] for _ in range(batch)]; b_act=[[] for _ in range(batch)]
        b_lp=[[] for _ in range(batch)];  b_rew=[[] for _ in range(batch)]
        for i in range(min(len(designs), batch)):
            cur = designs[i]; cur_obj = objs[i]; stop=False; guard=0
            while not stop and guard < 200 and NFE < budget + batch*4:
                guard += 1
                vlp, vidx, plp, pidx, slp, sdec = actor.sample_action(
                    torch.tensor(cur, dtype=torch.float32).to(dev))
                nd = list(cur); nd[vidx.item()] = pidx.item()
                nobjv, ncon, ncv = prob.evaluate(nd); NFE += 1
                all_des.append(nd); all_obj.append(nobjv); all_con.append(ncon)
                if not ncon: pass
                r = step_reward(cfg, cur_obj, nobjv, ncon, min_mask, archive, to_hv)
                b_act[i].append([vidx.item(), pidx.item(), sdec.item()])
                b_lp[i].append([vlp.item(), plp.item(), slp.item()])
                b_rew[i].append(r)
                b_obs[i].append(list(nd) if cfg['obs_mode'] == 'design'
                                else list(nd) + list(nobjv) + list(ncv))
                cur, cur_obj = nd, nobjv
                if sdec.item() == 1: stop = True
        rew_hist.append(float(np.mean([np.sum(r) for r in b_rew if r])))

        cvals=[[] for _ in range(batch)]
        with torch.no_grad():
            for i in range(batch):
                if not b_obs[i]: continue
                cvals[i] = np.array(critic.sample_critic(
                    torch.tensor([b_obs[i]], dtype=torch.float32).to(dev)).float().tolist())
        idx = [i for i in range(batch) if b_obs[i]]
        adv, ret = calculate_gae([b_rew[i] for i in idx], [cvals[i] for i in idx],
                                 cfg['gamma'], cfg['lam'])
        O=[torch.tensor(b_obs[i],dtype=torch.float32,device=dev) for i in idx]
        A=[torch.tensor(b_act[i],dtype=torch.float32,device=dev) for i in idx]
        L=[torch.tensor(b_lp[i],dtype=torch.float32,device=dev) for i in idx]
        AD=[torch.tensor(a,dtype=torch.float32,device=dev) for a in adv]
        RT=[torch.tensor(np.asarray(r).copy(),dtype=torch.float32,device=dev) for r in ret]
        for _ in range(cfg['update_iterations']):
            _, kl = actor.ppo_update(O,A,L,AD)
            if kl > cfg['target_kl']: break
        for _ in range(cfg['update_iterations']):
            critic.ppo_update(O,RT)

    all_des=np.array(all_des); all_obj=np.array(all_obj); all_con=np.array(all_con)
    rv = mo*np.ones(nobj); rv[~min_mask[:nobj]]=0.0; all_obj[all_con]=rv
    norm = all_obj/mo; hv_obj = deepcopy(norm)
    for i in range(nobj):
        if omm[i].lower()=='max': hv_obj[:,i]=1-norm[:,i]
    if cfg.get('save_actor'):
        os.makedirs(os.path.dirname(cfg['save_actor']), exist_ok=True)
        # atomic: workers poll for these files as a barrier, so a half-written
        # checkpoint must never be visible under its final name
        tmp = cfg['save_actor'] + '.tmp'
        torch.save(actor.state_dict(), tmp)
        os.replace(tmp, cfg['save_actor'])

    _,_,hvs = track_pareto_progress(hv_obj, all_des, all_con, hv_ind)
    return dict(hv_curve=[float(h) for h in hvs], hv=float(hvs[-1]), NFE=len(hvs),
                reward=rew_hist, wall=time.perf_counter()-t0,
                valid_frac=float((~all_con).mean()))
