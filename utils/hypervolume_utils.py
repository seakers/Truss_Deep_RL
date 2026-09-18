import numpy as np

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    

def track_pareto_progress(hv_obj, all_des, all_constraints, hv_ind):
    """
    Pareto front and hypervolume after every evaluation.

    Tests each new point against the current front rather than replaying the whole
    history, which makes this linear instead of quadratic. Entries are shared by
    reference on steps where the front does not change.
    """
    hv_obj = np.asarray(hv_obj)
    all_des = np.asarray(all_des)

    # A negative coordinate means the design beat max_values on that axis, which makes
    # pymoo integrate volume outside the unit cube and return scores above 1.0. Clipping
    # is a workaround; the reference itself needs rebuilding.
    if (hv_obj < 0).any():
        hv_obj = np.clip(hv_obj, 0.0, None)

    front_pts = None
    front_idx = None
    cur_obj = cur_des = None
    cur_hv = 0.0

    pareto_front_des, pareto_front_obj, hypervolumes = [], [], []

    for i in range(len(hv_obj)):
        # Feasible points only; the old guard let an infeasible first design seed the front.
        if not all_constraints[i]:
            point = hv_obj[i]
            if front_pts is None:
                front_pts, front_idx, changed = point[None, :], np.array([i]), True
            elif np.any(np.all(front_pts <= point, axis=1)):
                changed = False
            else:
                keep = ~np.all(point <= front_pts, axis=1)
                front_pts = np.vstack([front_pts[keep], point])
                front_idx = np.append(front_idx[keep], i)
                changed = True

            if changed:
                cur_obj = front_pts
                cur_des = all_des[front_idx]
                cur_hv = hv_ind(front_pts)

        pareto_front_obj.append(cur_obj)
        pareto_front_des.append(cur_des)
        hypervolumes.append(cur_hv)

    return pareto_front_des, pareto_front_obj, hypervolumes


class HypervolumeGrid:

    def __init__(self,refPoint):
        if isinstance(refPoint,list):
            refPoint = np.array(refPoint)
        self.refPoint = refPoint
        self.initializeGrid()

    def initializeGrid(self):
        """
        Function to initialize the hypervolume grid with the reference point
        """
        self.gridSize = 201 # maybe change this to have different resolution closer to the best point
        self.HVMax = np.prod(self.refPoint)

        exp = np.arange(len(self.refPoint)-1,-1,-1)
        self.mult = np.power(self.gridSize,exp)

        self.numPoints = (self.gridSize)**len(self.refPoint)
        self.dominated = np.zeros(int(self.numPoints), dtype=bool)

        self.paretoFrontPoint = np.array([])
        self.paretoFrontSolution = np.array([])

    def updateHV(self,point,solution):
        """
        Function to update the hypervolume grid and pareto front with a new point and index. 
        Index for each call should be unique and able to reference the solution.
        """
        if isinstance(point,list):
            point = np.array(point)
        if isinstance(solution,list):
            solution = np.array(solution)
        pointCalc = self.refPoint - point
        gridPoint = np.floor(pointCalc/self.refPoint * (self.gridSize-1))
        pointIdx = int(np.dot(gridPoint,self.mult))

        if not self.dominated[pointIdx]:
            dominatedInd = np.arange(self.numPoints,dtype=int)
            newDominated = np.ones(self.numPoints, dtype=bool)
            for dim in range(len(self.refPoint)):
                dimList = np.floor_divide(np.mod(dominatedInd,self.mult[dim]*self.gridSize,dtype=int),self.mult[dim],dtype=int)
                fitDim = dimList <= gridPoint[dim]
                newDominated = np.logical_and(newDominated,fitDim, dtype=bool)
            
            self.updateParetoFront(point,solution)

            self.dominated = np.logical_or(self.dominated, newDominated, dtype=bool)

    def updateParetoFront(self,newPoint,solution):
        """
        Add a new cost to the pareto front if it is not dominated
        """
        isDominated = False
        if self.paretoFrontPoint.size:
            isDominated = np.any(np.all(self.paretoFrontPoint < newPoint, axis = 1))
            newDominated = np.all(self.paretoFrontPoint > newPoint, axis = 1)

            # remove dominated points from pareto front (points stay as ndarray)
            self.paretoFrontPoint = self.paretoFrontPoint[~newDominated]

            # convert existing solutions to a Python list (handle empty ndarray or existing list)
            if isinstance(self.paretoFrontSolution, np.ndarray):
                if self.paretoFrontSolution.size == 0:
                    sol_list = []
                else:
                    sol_list = [s.tolist() if isinstance(s, np.ndarray) else s for s in self.paretoFrontSolution]
            else:
                sol_list = list(self.paretoFrontSolution)

            # filter solutions by the mask
            sol_list = [s for i, s in enumerate(sol_list) if not newDominated[i]]
            self.paretoFrontSolution = sol_list

        if isDominated == False:
            # append new Pareto point (keep as ndarray)
            self.paretoFrontPoint = (
                np.vstack([self.paretoFrontPoint, newPoint])
                if self.paretoFrontPoint.size else np.array([newPoint])
            )

            # append new solution into the Python list (preserve nested-list shape)
            new_sol = solution.tolist() if isinstance(solution, np.ndarray) else solution
            if isinstance(self.paretoFrontSolution, list):
                self.paretoFrontSolution.append(new_sol)
            else:
                # handle case where paretoFrontSolution was an empty ndarray
                self.paretoFrontSolution = [new_sol]

    def filterParetoFront(self, objInd, value):   
        """
        Filter the pareto front by the index of the objective and the desired value
        Ex: objInd = 1, value = 0.1 will return all points where the second objective is less than 0.1
        """
        if self.paretoFrontSolution.size:
            mask = self.paretoFrontPoint[:,objInd] < value
            self.paretoFrontPoint = self.paretoFrontPoint[mask]
            self.paretoFrontSolution = self.paretoFrontSolution[mask]

    def getHV(self):
        return np.sum(self.dominated)/self.numPoints * self.HVMax


class RunningParetoArchive:
    """
    Incremental Pareto front in normalised minimisation space.

    offer(point) returns (joined, delta_hv): whether the point entered the front and
    how much hypervolume it added.
    """

    def __init__(self, hv_ind):
        self.hv_ind = hv_ind
        self.pts = None
        self.hv = 0.0

    def offer(self, point):
        p = np.asarray(point, dtype=float)
        if self.pts is None:
            self.pts = p[None, :]
            new = float(self.hv_ind(self.pts))
            delta = new - self.hv
            self.hv = new
            return True, delta
        if np.any(np.all(self.pts <= p, axis=1)):      # dominated by the current front
            return False, 0.0
        keep = ~np.all(p <= self.pts, axis=1)          # drop what this point dominates
        self.pts = np.vstack([self.pts[keep], p])
        new = float(self.hv_ind(self.pts))
        delta = new - self.hv
        self.hv = new
        return True, delta
