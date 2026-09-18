import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

class TrussProblem:
    def __init__(self, load_factor=1.0):
        """
        Discrete 72-Bar Truss Benchmark
        Action Space: 72 discrete integers [0, 1, 2, 3]
        """
        self.n_elements = 72
        self.load_factor = load_factor
        self.n_nodes = 20
        
        # --- Discrete Options (Area in in^2) ---
        # self.area_map = {
        #     0: 1e-6, 1: 0.1, 2: 1.0, 3: 3.0
        # }
        self.area_map = np.linspace(0, 3.0, 50)
        self.area_map[0] = 1e-6  # To avoid zero area
        
        self.E = 10e6      
        self.rho = 0.1     
        self.stress_limit = 25000.0 
        self.disp_limit = 0.25      
        
        # --- Optimization Metadata ---
        self.num_objectives = 3
        self.num_constraints = 2
        # Objectives: Mass (min), Max Displacement (min), Frequency (max)
        self.objective_min_max = ['min', 'min', 'max', 'min', 'min'] 
        
        self.nodes, self.elements, self.fixed_nodes = self._generate_geometry()
        self.forces = self._generate_loads()
        self._precompute()
        
    def _generate_geometry(self):
        nodes = []
        dx = dy = dz = 120.0
        for z_level in range(5):
            z = z_level * dz
            nodes.append([dx, dy, z])    
            nodes.append([-dx, dy, z])   
            nodes.append([-dx, -dy, z])  
            nodes.append([dx, -dy, z])   
        nodes = np.array(nodes)
        
        elements = []
        for i in range(4): 
            base = 4 * i
            top = 4 * (i + 1)
            for j in range(4): elements.append([base+j, top+j])
            elements.extend([[base, top+1], [base+1, top], [base+1, top+2], [base+2, top+1],
                             [base+2, top+3], [base+3, top+2], [base+3, top], [base, top+3]])
            elements.extend([[top, top+1], [top+1, top+2], [top+2, top+3], [top+3, top]])
            elements.extend([[top, top+2], [top+1, top+3]])
        return nodes, np.array(elements), [0, 1, 2, 3]

    def _generate_loads(self):
        forces = np.zeros(self.n_nodes * 3)
        forces[16 * 3 + 2] = -5000.0 * self.load_factor
        return forces

    def _precompute(self):
        # Element geometry is identical for every design; only the areas change.
        vec = self.nodes[self.elements[:, 1]] - self.nodes[self.elements[:, 0]]
        self.lengths = np.linalg.norm(vec, axis=1)
        self.dirs = vec / self.lengths[:, None]

        n_el = len(self.elements)
        n_dof = self.n_nodes * 3

        T = np.zeros((n_el, 2, 6))
        T[:, 0, :3] = self.dirs
        T[:, 1, 3:] = self.dirs
        k_local = np.array([[1.0, -1.0], [-1.0, 1.0]])
        self.k_shape = np.einsum('eji,jk,ekl->eil', T, k_local, T).reshape(n_el, 36)

        self.dof_map = np.stack([
            self.elements[:, 0] * 3, self.elements[:, 0] * 3 + 1, self.elements[:, 0] * 3 + 2,
            self.elements[:, 1] * 3, self.elements[:, 1] * 3 + 1, self.elements[:, 1] * 3 + 2,
        ], axis=1)
        self.scatter_idx = (np.repeat(self.dof_map, 6, axis=1).ravel() * n_dof
                            + np.tile(self.dof_map, (1, 6)).ravel())
        self.mass_dofs = self.dof_map.ravel()

        fixed_dofs = [n * 3 + i for n in self.fixed_nodes for i in range(3)]
        self.free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)
        self.forces_free = self.forces[self.free_dofs]
        self.area_lut = np.asarray(self.area_map)

    def _assemble(self, areas):
        n_dof = self.n_nodes * 3
        k_scale = self.E * areas / self.lengths
        K = np.bincount(self.scatter_idx,
                        weights=(self.k_shape * k_scale[:, None]).ravel(),
                        minlength=n_dof * n_dof).reshape(n_dof, n_dof)
        m_half = self.rho * areas * self.lengths / 2.0
        M_diag = np.bincount(self.mass_dofs, weights=np.repeat(m_half, 6), minlength=n_dof)
        return K, M_diag

    def _displace(self, K_reduced):
        u_full = np.zeros(self.n_nodes * 3)
        u_full[self.free_dofs] = np.linalg.solve(K_reduced, self.forces_free)
        return u_full

    def _axial_stress(self, u_full):
        du = u_full[self.dof_map[:, 3:]] - u_full[self.dof_map[:, :3]]
        delta_l = np.einsum('ij,ij->i', du, self.dirs)
        return self.E * delta_l / self.lengths

    def solve(self, discrete_actions):
        design = np.asarray(discrete_actions)
        areas = self.area_lut[design]
        K_global, M_diag = self._assemble(areas)
        K_reduced = K_global[np.ix_(self.free_dofs, self.free_dofs)]

        try:
            u_full = self._displace(K_reduced)
        except np.linalg.LinAlgError:
            return {'valid': False, 'reason': 'Singular'}

        mass = float(np.sum(self.rho * areas * self.lengths))

        stress = self._axial_stress(u_full)
        active = design > 0
        stress_viol = float(np.sum(np.clip(np.abs(stress) - self.stress_limit, 0.0, None)[active]))

        max_disp = float(np.max(np.abs(u_full)))
        disp_viol = max(0.0, max_disp - self.disp_limit)

        try:
            vals, _ = eigh(K_reduced, np.diag(M_diag[self.free_dofs]), subset_by_index=[0, 0])
            frequency = float(np.sqrt(vals[0]) / (2 * np.pi)) if vals[0] > 0 else 0.0
        except Exception:
            frequency = 0.0

        # Per-bar utilisation: |stress| / stress_limit, signed by tension/compression.
        # Zeroed for removed bars, and clipped so a near-singular structure cannot feed
        # huge values into a network.
        util = np.clip(stress / self.stress_limit, -3.0, 3.0)
        util[~active] = 0.0

        return { # transformation of large scaling objectives and constraints
            'mass': mass,
            'max_disp': max_disp/(max_disp + 1),
            'frequency': frequency,
            'stress_viol': stress_viol/(stress_viol + 1),
            'disp_viol': disp_viol/(disp_viol + 1),
            'valid': (stress_viol == 0 and disp_viol == 0),
            'utilisation': util,
        }

    def evaluate(self, design, return_utilisation=False):
        """Returns (objectives, is_constrained, constraints).

        With return_utilisation=True a fourth element is appended: the per-bar
        utilisation vector (length n_elements). It is computed by the same solve either
        way, so asking for it costs nothing extra. Existing callers are unaffected.
        """
        res = self.solve(design)

        if not res.get('valid', False) and 'mass' not in res:
            # singular structure: solve() bailed out early
            objectives = [float(np.sum(self.rho * self.area_lut[np.asarray(design)]
                                       * self.lengths)), 1.0, 0.0]
            constraints = [1.0, 1.0]
            if return_utilisation:
                return objectives, True, constraints, np.zeros(self.n_elements)
            return objectives, True, constraints

        # Objectives: [Mass, MaxDisp, Frequency] (All Positive)
        objectives = [res['mass'], res['max_disp'], res['frequency']]
        constraints = [res['stress_viol'], res['disp_viol']]
        is_constrained = (res['stress_viol'] > 1e-6 or res['disp_viol'] > 1e-6)

        if return_utilisation:
            return objectives, is_constrained, constraints, res['utilisation']
        return objectives, is_constrained, constraints

    def render(self, discrete_actions, save_name, deformation_scale=100.0, show_constraints=True):
        """
        Visualizes the truss structure.
        """
        res = self.solve(discrete_actions)
        
        areas = self.area_lut[np.asarray(discrete_actions)]

        if not res['valid']:
             print(f"Cannot visualize {save_name}: Structure Invalid.")
             return

        K_reduced = self._assemble(areas)[0][np.ix_(self.free_dofs, self.free_dofs)]
        u_full = self._displace(K_reduced)
        element_stresses = self._axial_stress(u_full)

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('jet')
        norm = colors.Normalize(vmin=0, vmax=self.stress_limit)
        
        for idx, (n1, n2) in enumerate(self.elements):
            action = discrete_actions[idx]
            if action == 0: continue 
            
            p1, p2 = self.nodes[n1], self.nodes[n2]
            d1 = u_full[n1*3:n1*3+3] * deformation_scale
            d2 = u_full[n2*3:n2*3+3] * deformation_scale
            coord_1 = p1 + d1
            coord_2 = p2 + d2
            
            stress_val = abs(element_stresses[idx])
            color = cmap(norm(stress_val))
            linewidth = action * 0.1 
            ax.plot([coord_1[0], coord_2[0]], [coord_1[1], coord_2[1]], [coord_1[2], coord_2[2]], 
                    color=color, linewidth=linewidth, alpha=0.9)
            
        deformed_nodes = self.nodes + u_full.reshape(-1, 3) * deformation_scale
        ax.scatter(deformed_nodes[:,0], deformed_nodes[:,1], deformed_nodes[:,2], c='black', s=20)

        ax.set_title(f'{save_name}\nMass: {res["mass"]:.1f} | Freq: {res["frequency"]:.1f} Hz')
        
        # Scaling
        max_range = np.array([self.nodes[:,0].max()-self.nodes[:,0].min(), 
                              self.nodes[:,1].max()-self.nodes[:,1].min(), 
                              self.nodes[:,2].max()-self.nodes[:,2].min()]).max() / 2.0
        mid_x = (self.nodes[:,0].max()+self.nodes[:,0].min()) * 0.5
        mid_y = (self.nodes[:,1].max()+self.nodes[:,1].min()) * 0.5
        mid_z = (self.nodes[:,2].max()+self.nodes[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array([])
        plt.colorbar(m, ax=ax, shrink=0.5, pad=0.1, label='Stress (psi)')
        
        plt.savefig(save_name + ".png")
        plt.close()