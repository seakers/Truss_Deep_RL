import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

class TrussProblem:
    def __init__(self):
        """
        Discrete 72-Bar Truss Benchmark
        Action Space: 72 discrete integers [0, 1, 2, 3]
        """
        self.n_elements = 72
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
        forces[16 * 3 + 2] = -5000.0
        return forces

    def solve(self, discrete_actions):
        areas = np.array([self.area_map[a] for a in discrete_actions])
        n_dof = self.n_nodes * 3
        K_global = np.zeros((n_dof, n_dof))
        M_global = np.zeros((n_dof, n_dof))
        
        for idx, (n1, n2) in enumerate(self.elements):
            p1, p2 = self.nodes[n1], self.nodes[n2]
            L = np.linalg.norm(p2 - p1)
            cx, cy, cz = (p2 - p1) / L
            A = areas[idx]
            k_val = (self.E * A / L)
            T = np.array([[cx, cy, cz, 0, 0, 0], [0, 0, 0, cx, cy, cz]])
            k_local = np.array([[1, -1], [-1, 1]]) * k_val
            k_elem = T.T @ k_local @ T
            indices = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
            for r in range(6):
                for c in range(6):
                    K_global[indices[r], indices[c]] += k_elem[r, c]

            m_elem = self.rho * A * L
            for i in range(3):
                M_global[n1*3+i, n1*3+i] += m_elem / 2.0
                M_global[n2*3+i, n2*3+i] += m_elem / 2.0

        fixed_dofs = []
        for n in self.fixed_nodes: fixed_dofs.extend([n*3, n*3+1, n*3+2])
        free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
        
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        F_reduced = self.forces[free_dofs]
        M_reduced = M_global[np.ix_(free_dofs, free_dofs)]
        
        try:
            u_reduced = spsolve(coo_matrix(K_reduced), F_reduced)
        except:
            return {'valid': False, 'reason': 'Singular'}

        u_full = np.zeros(n_dof)
        u_full[free_dofs] = u_reduced
        
        mass = 0
        stress_viol = 0
        for idx, (n1, n2) in enumerate(self.elements):
            p1, p2 = self.nodes[n1], self.nodes[n2]
            L = np.linalg.norm(p2 - p1)
            mass += self.rho * areas[idx] * L
            u_el = np.concatenate([u_full[n1*3:n1*3+3], u_full[n2*3:n2*3+3]])
            vec = (p2 - p1) / L
            delta_l = np.dot(u_el[3:] - u_el[:3], vec)
            stress = (self.E * delta_l) / L
            if discrete_actions[idx] > 0 and abs(stress) > self.stress_limit:
                stress_viol += (abs(stress) - self.stress_limit)

        max_disp = np.max(np.abs(u_full))
        disp_viol = max(0, max_disp - self.disp_limit)

        try:
            vals, _ = eigh(K_reduced, M_reduced, subset_by_index=[0, 0])
            frequency = np.sqrt(vals[0]) / (2 * np.pi) if vals[0] > 0 else 0.0
        except:
            frequency = 0.0

        return { # transformation of large scaling objectives and constraints
            'mass': mass,
            'max_disp': max_disp/(max_disp + 1),
            'frequency': frequency,
            'stress_viol': stress_viol/(stress_viol + 1),
            'disp_viol': disp_viol/(disp_viol + 1),
            'valid': (stress_viol == 0 and disp_viol == 0)
        }

    def evaluate(self, design):
        res = self.solve(design)
            
        # Objectives: [Mass, MaxDisp, Frequency] (All Positive)
        objectives = [res['mass'], res['max_disp'], res['frequency']]
        constraints = [res['stress_viol'], res['disp_viol']]
        is_constrained = (res['stress_viol'] > 1e-6 or res['disp_viol'] > 1e-6)
        
        return objectives, is_constrained, constraints

    def render(self, discrete_actions, save_name, deformation_scale=100.0, show_constraints=True):
        """
        Visualizes the truss structure.
        """
        res = self.solve(discrete_actions)
        
        # Helper to re-run simple static solve for plotting vectors
        # (Copying the solve logic here strictly for visualization data extraction)
        areas = np.array([self.area_map[a] for a in discrete_actions])
        n_dof = self.n_nodes * 3
        K_global = np.zeros((n_dof, n_dof))
        for idx, (n1, n2) in enumerate(self.elements):
            p1, p2 = self.nodes[n1], self.nodes[n2]
            L = np.linalg.norm(p2 - p1)
            cx, cy, cz = (p2 - p1) / L
            A = areas[idx]
            k_val = (self.E * A / L)
            T = np.array([[cx, cy, cz, 0, 0, 0], [0, 0, 0, cx, cy, cz]])
            k_local = np.array([[1, -1], [-1, 1]]) * k_val
            k_elem = T.T @ k_local @ T
            indices = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
            for r in range(6):
                for c in range(6):
                    K_global[indices[r], indices[c]] += k_elem[r, c]
        
        fixed_dofs = []
        for n in self.fixed_nodes: fixed_dofs.extend([n*3, n*3+1, n*3+2])
        free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
        
        if not res['valid']:
             print(f"Cannot visualize {save_name}: Structure Invalid.")
             return

        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        F_reduced = self.forces[free_dofs]
        u_reduced = spsolve(coo_matrix(K_reduced), F_reduced)
        u_full = np.zeros(n_dof)
        u_full[free_dofs] = u_reduced
        
        element_stresses = []
        for idx, (n1, n2) in enumerate(self.elements):
            p1, p2 = self.nodes[n1], self.nodes[n2]
            L = np.linalg.norm(p2 - p1)
            u_el = np.concatenate([u_full[n1*3:n1*3+3], u_full[n2*3:n2*3+3]])
            vec = (p2 - p1) / L
            delta_l = np.dot(u_el[3:] - u_el[:3], vec)
            stress = (self.E * delta_l) / L
            element_stresses.append(stress)
        element_stresses = np.array(element_stresses)

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