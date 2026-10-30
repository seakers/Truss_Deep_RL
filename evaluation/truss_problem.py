import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import solve as linsolve
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors


TOP_NODES = [16, 17, 18, 19]

class TrussProblem:
    def __init__(self, load_factor=1.0, overlap_constraint=None, load_node=16): #set overlap to none to force an error when not specified
        """
        Discrete 72-Bar Truss Benchmark.

        load_node: node id receiving the downward point load. Must be one of
                   TOP_NODES. Default 16 preserves the original benchmark.
        """
        if overlap_constraint is None:
            raise ValueError("TrussProblem requires overlap_constraint to be specified (True/False).")
        
        self.n_elements = 72
        self.load_factor = load_factor
        self.load_node = load_node          # NEW
        self.n_nodes = 20
        self.overlap_constraint = overlap_constraint

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

        self.overlap_pairs = self._find_overlap_pairs() if overlap_constraint else []
        self.design_space, self._gene_layout = self._build_design_space()
        self._design_space_template = self.design_space        # physical genes (len varies by mode)

        # unique variable types: one 'Area' always; add 'Sel' only when pairs exist
        n_areas = len(self.area_map)
        self.unique_design_space = [{'type': 'discrete', 'range': list(range(n_areas)), 'name': 'Area'}]
        if self.overlap_pairs:
            self.unique_design_space.insert(0, {'type': 'discrete', 'range': [0, 1], 'name': 'Sel'})

        
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
        forces[self.load_node * 3 + 2] = -5000.0 * self.load_factor   # was 16*3+2
        return forces


    def _find_overlap_pairs(self):
        """Pairs of elements that cross: no shared node, segments intersect
        at an interior point of both. Fixed geometry -> computed once."""
        pairs = []
        n = len(self.elements)
        for i in range(n):
            a, b = self.nodes[self.elements[i]]
            for j in range(i + 1, n):
                if set(self.elements[i]) & set(self.elements[j]):
                    continue
                c, d = self.nodes[self.elements[j]]
                if self._segments_cross(a, b, c, d):
                    pairs.append((i, j))
        return pairs


    def _segments_cross(self, p1, p2, p3, p4, tol=1e-6):
        """True if segment p1p2 and segment p3p4 intersect at an interior point of both."""
        d1, d2 = p2 - p1, p4 - p3
        r = p3 - p1
        A = np.column_stack([d1, -d2])
        ts, _, rank, _ = np.linalg.lstsq(A, r, rcond=None)
        if rank < 2:
            return False  # parallel / degenerate
        t, s = ts
        gap = np.linalg.norm((p1 + t * d1) - (p3 + s * d2))
        return (tol < t < 1 - tol) and (tol < s < 1 - tol) and gap < tol


    def _build_design_space(self):
        """design_space dicts + a layout to decode genes -> length-72 area indices.
        layout entries: ('area', elem) or ('pair', elem_i, elem_j)."""
        n_areas = len(self.area_map)
        if not self.overlap_constraint:
            ds = [{'type': 'discrete', 'range': list(range(n_areas)),
                'name': 'Area'}] * self.n_elements
            return ds, [('area', e) for e in range(self.n_elements)]

        paired = {e for pair in self.overlap_pairs for e in pair}
        ds, layout = [], []
        for (i, j) in self.overlap_pairs:
            ds.append({'type': 'discrete', 'range': [0, 1], 'name': 'Sel'})
            ds.append({'type': 'discrete', 'range': list(range(n_areas)),
                    'name': 'Area'})
            layout.append(('pair', i, j))
        for e in range(self.n_elements):
            if e not in paired:
                ds.append({'type': 'discrete', 'range': list(range(n_areas)),
                        'name': 'Area'})
                layout.append(('area', e))
        return ds, layout


    def _decode(self, design):
        """Map a design-space gene vector to length-72 area-map indices.
        Deselected member of a pair stays at index 0 (~zero area)."""
        area_idx = np.zeros(self.n_elements, dtype=int)
        g = 0
        for entry in self._gene_layout:
            if entry[0] == 'area':
                _, e = entry
                area_idx[e] = int(design[g]); g += 1
            else:
                _, i, j = entry
                selector = int(design[g]); g += 1
                chosen = i if selector == 0 else j
                area_idx[chosen] = int(design[g]); g += 1
        return area_idx

    def solve(self, design):
        discrete_actions = self._decode(design)
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
            idx = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
            K_global[np.ix_(idx, idx)] += k_elem

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
            u_reduced = linsolve(K_reduced, F_reduced, assume_a='pos')
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
            vals = eigh(K_reduced, M_reduced, subset_by_index=[0, 0],
                        eigvals_only=True)
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

    def render(self, design, save_name, deformation_scale=100.0, show_constraints=True):
        """Visualizes the truss structure (HTML/Plotly)."""
        from utils.visualization import render_truss
        render_truss(self, design, save_name, deformation_scale, show_constraints)