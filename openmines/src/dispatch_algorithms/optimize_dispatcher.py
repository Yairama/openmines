from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite
from gurobipy import Model, GRB, quicksum  # Replace with pulp/ortools if gurobi is unavailable
import random

class OptimizeDispatcher(BaseDispatcher):
    def __init__(self, max_local_search_iter=50):
        super().__init__()
        self.name = "OptimizeDispatcher"
        self.max_local_search_iter = max_local_search_iter
        self.solution = {}  # { truck_name: (chosen_load_site_name, chosen_dump_site_name) }

    def compute_solution(self, mine:"Mine"):
        """
        Use MILP to obtain an initial assignment and then refine it with a local search.
        Produces self.solution = {truck_name -> (load_site_name, dump_site_name)}.
        """
        # Skip recomputation if a solution already exists
        if self.solution:
            return

        # ============= 1. Preparation: gather data =============

        trucks = mine.trucks
        load_sites = mine.load_sites
        dump_sites = mine.dump_sites

        # For each truck and load site/dump site pairing, approximate the cycle time cost.
        # Example: cost_{t,l,d} = T(charge->l) + load_time + T(l->d) + unload_time + T(d->l)
        # Note: we use a linear approximation (distance / speed * 60); feel free to refine it.
        cycle_time = {}
        for t_idx, truck in enumerate(trucks):
            for l_idx, load_site in enumerate(load_sites):
                # Travel time from the charging site to the load site
                dist_init = mine.road.charging_to_load[l_idx]
                time_init = dist_init / truck.truck_speed * 60
                # Approximate load time by averaging the shovels at this site
                load_time = 0.0
                if load_site.shovel_list:
                    avg_shovel_tons = sum(sv.shovel_tons for sv in load_site.shovel_list)/len(load_site.shovel_list)
                    avg_shovel_cycle = sum(sv.shovel_cycle_time for sv in load_site.shovel_list)/len(load_site.shovel_list)
                    load_time = (truck.truck_capacity / avg_shovel_tons) * avg_shovel_cycle
                for d_idx, dump_site in enumerate(dump_sites):
                    # Haul time from the load site to the dump site
                    dist_haul = mine.road.l2d_road_matrix[l_idx][d_idx]
                    time_haul = dist_haul / truck.truck_speed * 60
                    # Approximate unload time by averaging all dumpers
                    unload_time = 0.0
                    if dump_site.dumper_list:
                        avg_dumper_cycle = sum(dp.dump_time for dp in dump_site.dumper_list)/len(dump_site.dumper_list)
                        unload_time = avg_dumper_cycle
                    # Return trip from the dump site back to the load site
                    dist_unhaul = mine.road.d2l_road_matrix[d_idx][l_idx]  # Same matrix, l->d vs d->l
                    time_unhaul = dist_unhaul / truck.truck_speed * 60
                    # Total cycle time
                    total_cycle = time_init + load_time + time_haul + unload_time + time_unhaul
                    cycle_time[(t_idx, l_idx, d_idx)] = total_cycle

        # ============= 2. Build the MILP =============
        model = Model("MineTruckAssignment")
        model.setParam('OutputFlag', 0)  # Silence the solver log

        # Decision variable x_{t,l,d} ∈ {0,1}, indicates whether truck t picks (load_site l, dump_site d)
        x = {}
        for t_idx in range(len(trucks)):
            for l_idx in range(len(load_sites)):
                for d_idx in range(len(dump_sites)):
                    x[(t_idx, l_idx, d_idx)] = model.addVar(vtype=GRB.BINARY,
                                                            name=f"x_{t_idx}_{l_idx}_{d_idx}")

        # Constraint: each truck must be assigned to exactly one (l, d) pair
        for t_idx in range(len(trucks)):
            model.addConstr(quicksum(x[(t_idx, l_idx, d_idx)]
                                     for l_idx in range(len(load_sites))
                                     for d_idx in range(len(dump_sites))) == 1,
                            name=f"truck_{t_idx}_one_route")

        # Objective: minimize sum_{t,l,d} ( cycle_time_{t,l,d} * x_{t,l,d} )
        obj = quicksum(cycle_time[(t_idx, l_idx, d_idx)] * x[(t_idx, l_idx, d_idx)]
                       for t_idx in range(len(trucks))
                       for l_idx in range(len(load_sites))
                       for d_idx in range(len(dump_sites)))
        model.setObjective(obj, GRB.MINIMIZE)

        # Solve the MILP
        model.optimize()

        # ============= 3. Interpret the MILP solution =============
        assignment = {}  # truck_idx -> (l_idx, d_idx)
        for t_idx in range(len(trucks)):
            for l_idx in range(len(load_sites)):
                for d_idx in range(len(dump_sites)):
                    if x[(t_idx, l_idx, d_idx)].X > 0.5:  # Truck t_idx selected (l_idx, d_idx)
                        assignment[t_idx] = (l_idx, d_idx)
                        break

        # ============= 4. Convert to the self.solution structure =============
        for t_idx, (l_idx, d_idx) in assignment.items():
            truck_name = trucks[t_idx].name
            load_site_name = load_sites[l_idx].name
            dump_site_name = dump_sites[d_idx].name
            self.solution[truck_name] = (load_site_name, dump_site_name)

        # ============= 5. Optional local search =============
        # Randomly swap the (load_site, dump_site) assignments for two trucks to seek improvements.
        best_obj_val = self._evaluate_solution(self.solution, cycle_time, trucks, load_sites, dump_sites)
        for _ in range(self.max_local_search_iter):
            # Pick two trucks at random
            tA, tB = random.sample(range(len(trucks)), 2)
            truckA_name = trucks[tA].name
            truckB_name = trucks[tB].name
            oldA = self.solution[truckA_name]
            oldB = self.solution[truckB_name]
            # Swap their assignments
            self.solution[truckA_name] = oldB
            self.solution[truckB_name] = oldA
            new_obj_val = self._evaluate_solution(self.solution, cycle_time, trucks, load_sites, dump_sites)
            if new_obj_val + 1e-6 < best_obj_val:
                best_obj_val = new_obj_val
            else:
                # Revert if there is no improvement
                self.solution[truckA_name] = oldA
                self.solution[truckB_name] = oldB

        print(f"[OptimizeDispatcher] MILP + LocalSearch done, final objective = {best_obj_val:.2f}")

    def _evaluate_solution(self, solution, cycle_time, trucks, load_sites, dump_sites):
        """Helper to compute the total cycle time for a given mapping of trucks to (load_site, dump_site)."""
        total = 0.0
        # Build reverse lookups for site indices
        l_name_to_idx = {ls.name: i for i, ls in enumerate(load_sites)}
        d_name_to_idx = {ds.name: i for i, ds in enumerate(dump_sites)}
        for t_idx, truck in enumerate(trucks):
            truck_name = truck.name
            if truck_name not in solution:
                continue
            ls_name, ds_name = solution[truck_name]
            l_idx = l_name_to_idx[ls_name]
            d_idx = d_name_to_idx[ds_name]
            total += cycle_time[(t_idx, l_idx, d_idx)]
        return total

    # ========== The following handlers supply instructions to Truck.run() ==========

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        Tell a newly dispatched truck which load site to visit first.
        """
        if not self.solution:
            self.compute_solution(mine)
        load_sites = {ls.name: i for i, ls in enumerate(mine.load_sites)}

        # Use the solution's mapping truck->(load_site_name, dump_site_name) to return the site index
        load_site_name, _ = self.solution[truck.name]
        return load_sites[load_site_name]

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        After loading, indicate which dump site the truck should visit.
        """
        if not self.solution:
            self.compute_solution(mine)
        dump_sites = {ds.name: i for i, ds in enumerate(mine.dump_sites)}

        # Use the solution mapping to return the dump site index
        _, dump_site_name = self.solution[truck.name]
        return dump_sites[dump_site_name]

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        After unloading, indicate which load site the truck should return to.
        """
        if not self.solution:
            self.compute_solution(mine)
        load_sites = {ls.name: i for i, ls in enumerate(mine.load_sites)}

        load_site_name, _ = self.solution[truck.name]
        return load_sites[load_site_name]
