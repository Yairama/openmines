import simpy

from openmines.src.utils.event import EventPool


class ParkingLot:
    def __init__(self,name:str,position:tuple):
        self.name = name
        self.position = position
        self.queue_status = dict()
        self.resources = []
        self.res_objs = []
        self.env = None

    def get_waiting_trucks(self, resource_list:list[simpy.Resource])->int:
        waiting_trucks = []
        for resource in resource_list:
            waiting_trucks += resource.queue
        return len(waiting_trucks)

    def update_queue_wait_status(self):
        # Get Reference
        res_objs = self.res_objs
        resources = self.resources
        env = self.env
        # Compute expected waiting times
        # Retrieve the information stored in custom load/dump requests
        for index, res_obj in enumerate(res_objs):  # Iterate over each shovel or dumper resource
            res_name = res_obj.name
            # If the resource is a shovel, derive its queue waiting time estimate
            if res_obj.type == "shovel":
                shovel_name = res_obj.name
                shovel_tons = res_obj.shovel_tons
                shovel_cycle_time = res_obj.shovel_cycle_time
                shovel_queue_wait_total_time = 0
                shovel_last_service_time = max(res_obj.last_service_time,res_obj.last_service_done_time)  # Most recent shovel service start
                for i, load_request in enumerate(resources[index].queue + resources[index].users):  # Walk through current loading requests
                    truck = load_request.truck
                    load_site = load_request.load_site
                    shovel_queue_wait_total_time += (truck.truck_capacity / res_obj.shovel_tons) * res_obj.shovel_cycle_time
                time_used_loading = env.now - shovel_last_service_time if resources[index].queue + resources[index].users else 0
                shovel_estimated_waiting_time = max(shovel_last_service_time + shovel_queue_wait_total_time - env.now, 0)  # Duration, not a timestamp
                res_obj.est_waiting_time = shovel_estimated_waiting_time  # Estimated waiting time for the shovel
                assert time_used_loading >= 0, "Time used for loading should be greater than 0"
            # If the resource is a dumper, derive its queue waiting time estimate
            if res_obj.type == "dumper":
                dumper_name = res_obj.name
                dumper_cycle_time = res_obj.dump_time
                dumper_queue_wait_total_time = 0
                dumper_last_service_time = res_obj.last_service_time  # Most recent dumper service time
                for i, dump_request in enumerate(resources[index].queue):  # Walk through pending dump requests
                    truck = dump_request.truck
                    dump_site = dump_request.dump_site
                    dumper_queue_wait_total_time += res_obj.dump_time
                time_used_dumping = env.now - dumper_last_service_time
                dumper_estimated_waiting_time = (dumper_queue_wait_total_time - time_used_dumping) if dumper_queue_wait_total_time > time_used_dumping else 0
                res_obj.est_waiting_time = dumper_estimated_waiting_time  # Estimated waiting time for the dumper
                assert time_used_dumping >= 0, "Time used for dumping should be greater than 0"

        # UPDATE LOAD_SITE AND DUMP_SITE(from the summary by each shovel and dumper, take the mini)
        if res_objs and res_objs[-1].type == "shovel":
            load_site = res_objs[-1].load_site
            load_site.estimated_queue_wait_time = min([shovel.est_waiting_time for shovel in res_objs])  # Average waiting time at the load site
            load_site.avg_queue_wait_time = sum([shovel.est_waiting_time for shovel in res_objs]) / len(res_objs)
        if res_objs and res_objs[-1].type == "dumper":
            dump_site = res_objs[-1].dump_site
            dump_site.estimated_queue_wait_time = min([dumper.est_waiting_time for dumper in res_objs])  # Average waiting time at the dump site
            dump_site.avg_queue_wait_time = sum([dumper.est_waiting_time for dumper in res_objs]) / len(res_objs)

    def monitor_resources(self, env, resources, res_objs, monitor_interval=1):
        """Monitor queue lengths for all parking-lot resources, down to individual shovels and dumpers."""
        self.res_objs = res_objs
        self.resources = resources
        self.env = env

        self.queue_status["total"] = dict()
        while True:
            # Record the current aggregate queue length
            all_queue_len = sum([len(resource.queue) for resource in resources])
            self.queue_status["total"][int(env.now)] = all_queue_len
            self.queue_status["total"]["cur_value"] = all_queue_len
            self.update_queue_wait_status()
            # Wait for the next monitoring interval
            yield env.timeout(monitor_interval)

    def monitor_resource(self, env, res_obj, resource, monitor_interval=1):
        """Monitor queue length for a single shovel or dumper resource.

        Custom LoadRequest/DumpRequest objects carry load_site/dump_site and truck information, which the
        parking lot uses to maintain per-resource statistics.

        res_obj: shovel or dumper
        resource: shovel or dumper resource
        """
        res_name = res_obj.name
        self.queue_status[res_name] = dict()
        while True:
            # Record the current queue length for the resource
            self.queue_status[res_name][int(env.now)] = len(resource.queue)
            self.queue_status[res_name]["cur_value"] = len(resource.queue)

            # Wait for the next monitoring interval
            yield env.timeout(monitor_interval)

class Shovel:
    def __init__(self, name:str, shovel_tons:float, shovel_cycle_time:float,position_offset:tuple=(0, 0.05)):
        self.name = name
        self.type = "shovel"
        self.load_site = None
        self.position = None
        self.position_offset = position_offset
        self.shovel_tons = shovel_tons  # bucket capacity of the shovel
        self.produced_tons = 0  # accumulated tonnage produced by the shovel
        self.service_count = 0  # number of shovel-vehicle cycles completed
        self.shovel_cycle_time = shovel_cycle_time  # time to load a truck once
        self.status = dict()  # status snapshots captured over time
        self.last_service_time = 0  # timestamp when the shovel last started loading a truck
        self.last_service_done_time = 0  # timestamp when the shovel last finished loading a truck
        self.est_waiting_time = 0  # estimated waiting time until the next truck
        # shovel breakdown
        self.event_pool = EventPool()
        self.last_breakdown_time = 0  # last breakdown timestamp
        self.repair = False  # whether the shovel is under repair

    def set_env(self, env:simpy.Environment):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)

    def monitor_status(self, env, monitor_interval=1):
        """Track shovel productivity and service counts over time."""
        while True:
            # Capture shovel metrics
            self.status[int(env.now)] = {
                "repair": self.repair,
                "produced_tons": self.produced_tons,
                "service_count": self.service_count,
            }
            # Wait for the next monitoring interval
            yield env.timeout(monitor_interval)


class LoadSite:
    def __init__(self, name:str, position:tuple):
        self.name = name
        self.position = position
        self.shovel_list = []
        self.parking_lot = None
        self.status = dict()  # the status of shovel
        self.produced_tons = 0  # total tonnage produced by the load site
        self.service_count = 0  # number of shovel-vehicle cycles at the load site
        self.service_ability_ratio = 0  # the ability of load site to serve trucks(0-1), the shovel may be breakdown
        self.estimated_queue_wait_time = 0  # the estimation of total waiting time for coming trucks in queue
        self.avg_queue_wait_time = 0  # the average waiting time for coming trucks in queue
        self.load_site_productivity = 0  # the productivity of load site
        # service_time = min service time
        self.last_service_time = 0  # last service start time
        self.last_service_done_time = 0  # last service completion time

    def update_service_time(self):
        self.last_service_time = min([shovel.last_service_time for shovel in self.shovel_list])
        self.last_service_done_time = min([shovel.last_service_done_time for shovel in self.shovel_list])

    def set_env(self, env:simpy.Environment):
        self.env = env
        for shovel in self.shovel_list:
            shovel.set_env(env)

    def monitor_status(self, env, monitor_interval=1):
        """Track load site productivity and service statistics."""
        while True:
            # Capture shovel metrics
            self.produced_tons = sum(shovel.produced_tons for shovel in self.shovel_list)
            self.service_count = sum(shovel.service_count for shovel in self.shovel_list)
            self.status[int(env.now)] = {
                "produced_tons": self.produced_tons,
                "service_count": self.service_count,
            }
            # Compute theoretical loading capacity
            load_site_productivity = sum(
                shovel.shovel_tons / shovel.shovel_cycle_time for shovel in self.shovel_list)
            self.load_site_productivity = load_site_productivity
            self.service_ability_ratio = sum((shovel.shovel_tons / shovel.shovel_cycle_time) * (0 if shovel.repair else 1) for shovel in self.shovel_list) / self.load_site_productivity
            # Wait for the next monitoring interval
            yield env.timeout(monitor_interval)


    def show_shovels(self):
        shovel_names = []
        for shovel in self.shovel_list:
            shovel_names.append(shovel.name)
        print(f'{self.name} has {shovel_names}')

    def add_shovel(self, shovel:Shovel):
        # Position the shovel relative to the load site; offset differs from dumpers
        shovel.position = tuple(a + b for a, b in zip(self.position, shovel.position_offset))
        # Keep a back-reference on the shovel
        shovel.load_site = self
        self.shovel_list.append(shovel)

    def add_parkinglot(self, position_offset, name: str = None):
        if name is None:
            name = f'{self.name}_parking_lot'
        park_position = tuple(a + b for a, b in zip(self.position, position_offset))
        self.parking_lot = ParkingLot(name=name, position=park_position)

    def get_available_shovel(self)->Shovel:
        """Return the first idle shovel using a simple greedy strategy.

        TODO: Expose shovels as independent units for decision making instead of using a greedy heuristic.
        """
        for shovel in self.shovel_list:
            if shovel.res.count == 0:
                return shovel
        # If none are idle, return the shovel with the shortest queue
        min_queue_shovel = min(self.shovel_list, key=lambda shovel: len(shovel.res.queue))
        return min_queue_shovel