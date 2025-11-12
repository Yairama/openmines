"""Core simulation object for the mine model.

Usage overview:
1. Instantiate :class:`Mine`.
2. Add :class:`LoadSite` (shovels), :class:`DumpSite` (dumpers),
   :class:`ChargingSite` (trucks), :class:`Road`, :class:`Dispatcher`, and other
   components.
3. Call :meth:`Mine.run` to start the simulation.
"""
import glob
import os
import time
from datetime import datetime

import numpy as np
import simpy,logging,math
from functools import reduce
from multiprocessing import Queue

from openmines.src.charging_site import ChargingSite
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite
from openmines.src.road import Road
from openmines.src.truck import Truck
from openmines.src.utils.logger import MineLogger, LOG_FILE_PATH
from openmines.src.utils.ticker import TickGenerator
from openmines.src.utils.event import EventPool

class Mine:
    def __init__(self, name:str, log_path=LOG_FILE_PATH, log_file_level=logging.DEBUG, log_console_level=logging.INFO, seed=42):
        self.env = simpy.Environment()
        self.name = name
        self.load_sites = []
        self.dump_sites = []
        self.trucks = []
        self.road = None
        self.dispatcher = None
        self.random_event_pool = EventPool()  # Random event pool
        # rl
        self.done = False
        # summary
        self.produce_tons = 0  # the produced tons of this dump site
        self.service_count = 0  # the number of shovel-vehicle cycle in this dump site
        self.status = dict()  # the status of shovel
        # Logger configuration
        # print(log_path)
        self.global_logger = MineLogger(log_path=log_path, file_level=log_file_level, console_level=log_console_level)
        self.mine_logger = self.global_logger.get_logger(name)

    def monitor_status(self, env, monitor_interval=1):
        """Monitor dump-site output, road conditions, and random events."""
        while True:
            # 1) Aggregate dump-site production and service statistics
            self.produce_tons = sum(dump_site.produce_tons for dump_site in self.dump_sites)
            self.service_count = sum(dump_site.service_count for dump_site in self.dump_sites)

            # Gather fleet-wide truck statistics
            working_truck_count = 0
            waiting_truck_count = 0
            load_unload_truck_count = 0
            moving_truck_count = 0
            repairing_truck_count = 0

            for truck in self.trucks:
                # Count trucks that are currently available for work
                if truck.status != "unrepairable":
                    working_truck_count += 1
                # Count trucks waiting in any queue
                if "waiting" in truck.status:
                    waiting_truck_count += 1
                # Count trucks that are loading or unloading
                if truck.status in {"loading", "unloading"}:
                    load_unload_truck_count += 1
                # Count trucks on the move
                if truck.status == "moving":
                    moving_truck_count += 1
                # Count trucks under repair
                if truck.status == "repairing":
                    repairing_truck_count += 1

            # Derive statistics for currently active random events
            road_jam_count = 0
            road_repair_count = 0
            truck_repair = 0
            truck_unrepairable = 0

            for event_key in self.random_event_pool.event_set.keys():
                event_type = self.random_event_pool.event_set[event_key].event_type
                if event_type == "RoadEvent:jam":
                    road_jam_count += 1
                if event_type == "RoadEvent:repair":
                    road_repair_count += 1
                if event_type == "TruckEvent:breakdown":
                    truck_repair += 1
                if event_type == "TruckEvent:unrepairable":
                    truck_unrepairable += 1

            random_event_count = road_jam_count + road_repair_count + truck_repair + truck_unrepairable

            self.status[int(env.now)] = {
                # Key performance indicators
                "produced_tons": self.produce_tons,
                "service_count": self.service_count,
                # Fleet statistics
                "truck_count": len(self.trucks),
                "working_truck_count": working_truck_count,
                "waiting_truck_count": waiting_truck_count,
                "load_unload_truck_count": load_unload_truck_count,
                "moving_truck_count": moving_truck_count,
                "repairing_truck_count": repairing_truck_count,
                # Random-event statistics
                "road_jam_count": road_jam_count,
                "road_repair_count": road_repair_count,
                "truck_repair": truck_repair,
                "truck_unrepairable": truck_unrepairable,
                "random_event_count": random_event_count
            }
            self.status["cur"] = self.status[int(env.now)]

            # 2) Update the aggregated road status information
            self.update_road_status()

            # 3) Sample potential truck breakdowns and repairs
            for truck in self.trucks:
                truck.sample_breakdown()

            # 4) Sample road congestion events based on current traffic
            # Iterate over each road segment configuration
            # Charging -> Load
            for i in range(self.road.load_site_num):
                charging_site = self.charging_site
                load_site = self.load_sites[i]
                self.road.road_jam_sampling(start=charging_site, end=load_site)

            # Load -> Dump
            for i in range(self.road.load_site_num):
                for j in range(self.road.dump_site_num):
                    load_site = self.load_sites[i]
                    dump_site = self.dump_sites[j]
                    self.road.road_jam_sampling(start=load_site, end=dump_site)

            # Dump -> Load
            for j in range(self.road.dump_site_num):
                for i in range(self.road.load_site_num):
                    load_site = self.load_sites[i]
                    dump_site = self.dump_sites[j]
                    self.road.road_jam_sampling(start=dump_site, end=load_site)

            yield env.timeout(monitor_interval)

    def update_road_status(self):
        cur_time = self.env.now
        road = self.road
        road_status = dict()

    # Optimization 1: precompute jam and repair counts keyed by (start_location, end_location)
        jam_events_dict = {}
        repair_events_dict = {}

        jam_events = self.random_event_pool.get_even_by_type("RoadEvent:jam")
        repair_events = self.random_event_pool.get_even_by_type("RoadEvent:repair")

        for jam_event in jam_events:
            start_loc = jam_event.info["start_location"]
            end_loc = jam_event.info["end_location"]
            if jam_event.info["start_time"] <= cur_time and jam_event.info["est_end_time"] >= cur_time:
                jam_events_dict[(start_loc, end_loc)] = jam_events_dict.get((start_loc, end_loc), 0) + 1

        for repair_event in repair_events:
            start_loc = repair_event.info["start_location"]
            end_loc = repair_event.info["end_location"]
            if repair_event.info["repair_start_time"] <= cur_time and repair_event.info["repair_end_time"] >= cur_time:
                repair_events_dict[(start_loc, end_loc)] = repair_events_dict.get((start_loc, end_loc), 0) + 1

    # Optimization 2: precompute how many trucks occupy each road segment
        truck_count_dict = {}
        for truck in self.trucks:
            if not truck.current_location or not truck.target_location:
                continue
            pair = (truck.current_location.name, truck.target_location.name)
            truck_count_dict[pair] = truck_count_dict.get(pair, 0) + 1

    # The loops below keep the original logic but reuse the precomputed jam/repair/truck dictionaries
        # 1. charging_site -> load_sites
        for i in range(self.road.load_site_num):
            charging_site_name = self.charging_site.name
            load_site_name = self.load_sites[i].name
            road_status_key = (charging_site_name, load_site_name)
            road_status[road_status_key] = {
                "truck_jam_count": jam_events_dict.get(road_status_key, 0),
                "repair_count": repair_events_dict.get(road_status_key, 0),
                "truck_count": truck_count_dict.get(road_status_key, 0)
            }

        # 2. load_sites -> dump_sites
        for i in range(self.road.load_site_num):
            for j in range(self.road.dump_site_num):
                load_site_name = self.load_sites[i].name
                dump_site_name = self.dump_sites[j].name
                road_status_key = (load_site_name, dump_site_name)
                road_status[road_status_key] = {
                    "truck_jam_count": jam_events_dict.get(road_status_key, 0),
                    "repair_count": repair_events_dict.get(road_status_key, 0),
                    "truck_count": truck_count_dict.get(road_status_key, 0)
                }

        # 3. dump_sites -> load_sites
        for j in range(self.road.dump_site_num):
            for i in range(self.road.load_site_num):
                dump_site_name = self.dump_sites[j].name
                load_site_name = self.load_sites[i].name
                road_status_key = (dump_site_name, load_site_name)
                road_status[road_status_key] = {
                    "truck_jam_count": jam_events_dict.get(road_status_key, 0),
                    "repair_count": repair_events_dict.get(road_status_key, 0),
                    "truck_count": truck_count_dict.get(road_status_key, 0)
                }

        self.road.road_status = road_status

    def add_load_site(self, load_site:LoadSite):
        load_site.set_env(self.env)
        self.load_sites.append(load_site)

    def add_charging_site(self, charging_site:ChargingSite):
        charging_site.set_mine(self)
        self.charging_site = charging_site
        self.trucks = charging_site.trucks
        for truck in self.trucks:
            truck.set_env(self)

    def add_dump_site(self, dump_site:DumpSite):
        dump_site.set_env(self.env)
        self.dump_sites.append(dump_site)

    def add_road(self, road:Road):
        assert road is not None, "road can not be None"
        road.set_env(self)
        self.road = road

    def add_dispatcher(self, dispatcher:BaseDispatcher):
        assert dispatcher is not None, "dispatcher can not be None"
        self.dispatcher = dispatcher

    def get_load_site_status(self):
        for load_site in self.load_sites:
            print(f'{load_site.name} has {len(load_site.res.users)} trucks')

    def get_dest_index_by_name(self,name:str):
        """
        Match load or dump sites by name and return their index.

        Index definition:
            None: no match
            0~N-1: load site indices
            N~N+M-1: dump site indices

        :param name:
        :return:
        """
        assert name is not None, "name can not be None"
        # Check load sites
        for i,loadsite in enumerate(self.load_sites):
            if loadsite.name == name:
                return i
        # Check dump sites
        for i,dumpsite in enumerate(self.dump_sites):
            if dumpsite.name == name:
                return i+len(self.load_sites)
        return None

    def get_dest_obj_by_index(self,index:int):
        """
        Retrieve the load or dump site object by its index value.

        :param index:
        :return:
        """
        assert index is not None, "index can not be None"
        if index < len(self.load_sites):
            return self.load_sites[index]
        elif index < len(self.load_sites) + len(self.dump_sites):
            return self.dump_sites[index - len(self.load_sites)]
        else:
            return None

    def get_dest_obj_by_name(self,name:str):
        """
        Return the load site, dump site, or charging site object that matches the given name.

        :param name:
        :return:
        """
        assert name is not None, "name can not be None"
        # Check load sites
        for loadsite in self.load_sites:
            if loadsite.name == name:
                return loadsite
        # Check dump sites
        for dumpsite in self.dump_sites:
            if dumpsite.name == name:
                return dumpsite
        # Check charging site
        if self.charging_site.name == name:
            return self.charging_site
        return None

    def get_service_vehicle_by_name(self,name:str):
        """
        Match shovels or dumpers by name and return the corresponding object.

        :param name:
        :return:
        """
        assert name is not None, "name can not be None"
        for loadsite in self.load_sites:
            for shovel in loadsite.shovel_list:
                if shovel.name == name:
                    return shovel
        for dumpsite in self.dump_sites:
            for dumper in dumpsite.dumper_list:
                if dumper.name == name:
                    return dumper
        return None

    def summary(self):
        """Record production metrics, matching factors, and wait times for each load and dump site after the simulation ends."""
        pass

    def start(self, total_time:float=60*8)->dict:
        """
        Entry point for simulating classical dispatch strategies, including RL inference runs.

        :param total_time:
        :return:
        """
        assert self.road is not None, "road can not be None"
        assert self.dispatcher is not None, "dispatcher can not be None"
        assert self.charging_site is not None, "charging_site can not be None"
        assert len(self.load_sites) > 0, "load_sites can not be empty"
        assert len(self.dump_sites) > 0, "dump_sites can not be empty"
        assert len(self.trucks) > 0, "trucks can not be empty"
        assert total_time > 0, "total_time can not be negative"
        self.total_time = total_time
        self.mine_logger.info(f"simulation started with dispatcher {self.dispatcher.__class__.__name__}")

        # start some monitor process for summary
        for load_site in self.load_sites:
            # Monitor parking lot queues
            self.env.process(load_site.parking_lot.monitor_resources(env=self.env,
                                                                    resources=[shovel.res for shovel in load_site.shovel_list],
                                                                     res_objs=load_site.shovel_list))
            for shovel in load_site.shovel_list:
                self.env.process(load_site.parking_lot.monitor_resource(env=self.env,res_obj=shovel,
                                                                        resource=shovel.res))
            # Monitor shovel throughput
            for shovel in load_site.shovel_list:
                self.env.process(shovel.monitor_status(env=self.env))
            # Monitor load site throughput and queue status
            self.env.process(load_site.monitor_status(env=self.env))

        for dump_site in self.dump_sites:
            # Monitor parking lot queues
            self.env.process(dump_site.parking_lot.monitor_resources(env=self.env,
                                                                    resources=[dumper.res for dumper in dump_site.dumper_list],
                                                                     res_objs=dump_site.dumper_list))
            for dumper in dump_site.dumper_list:
                self.env.process(dump_site.parking_lot.monitor_resource(env=self.env,res_obj=dumper,
                                                                        resource=dumper.res))
            # Monitor dump site throughput and queue status
            self.env.process(dump_site.monitor_status(env=self.env))
            # Monitor dumper throughput
            for dumper in dump_site.dumper_list:
                self.env.process(dumper.monitor_status(env=self.env))
        # Monitor mine-wide metrics
        self.env.process(self.monitor_status(env=self.env))

        # log in the truck as process
        for truck in self.trucks:
            self.env.process(truck.run())

        self.env.run(until=total_time)
        self.mine_logger.info(f"simulation finished with dispatcher {self.dispatcher.__class__.__name__}")
        self.summary()
        ticks = self.dump_frames(total_time=total_time)
        return ticks

    def start_rl(self, obs_queue:Queue, act_queue:Queue, reward_mode:str = "dense", total_time:float=60*8, ticks:bool=False)->dict:
        """
        Entry point for simulations that interact with RL algorithms.

        :param total_time:
        :return:
        """
        assert self.road is not None, "road can not be None"
        assert self.dispatcher is not None, "dispatcher can not be None"
        assert self.charging_site is not None, "charging_site can not be None"
        assert len(self.load_sites) > 0, "load_sites can not be empty"
        assert len(self.dump_sites) > 0, "dump_sites can not be empty"
        assert len(self.trucks) > 0, "trucks can not be empty"
        assert total_time > 0, "total_time can not be negative"
        self.total_time = total_time
        self.mine_logger.info("simulation started")
        # pass queue to dispatcher
        self.dispatcher.obs_queue = obs_queue
        self.dispatcher.act_queue = act_queue

        # start some monitor process for summary
        for load_site in self.load_sites:
            # Monitor parking lot queues
            self.env.process(load_site.parking_lot.monitor_resources(env=self.env,
                                                                     resources=[shovel.res for shovel in
                                                                                load_site.shovel_list],
                                                                     res_objs=load_site.shovel_list))
            for shovel in load_site.shovel_list:
                self.env.process(load_site.parking_lot.monitor_resource(env=self.env, res_obj=shovel,
                                                                        resource=shovel.res))
            # Monitor shovel throughput
            for shovel in load_site.shovel_list:
                self.env.process(shovel.monitor_status(env=self.env))
            # Monitor load site throughput and queue status
            self.env.process(load_site.monitor_status(env=self.env))

        for dump_site in self.dump_sites:
            # Monitor parking lot queues
            self.env.process(dump_site.parking_lot.monitor_resources(env=self.env,
                                                                     resources=[dumper.res for dumper in
                                                                                dump_site.dumper_list],
                                                                     res_objs=dump_site.dumper_list))
            for dumper in dump_site.dumper_list:
                self.env.process(dump_site.parking_lot.monitor_resource(env=self.env, res_obj=dumper,
                                                                        resource=dumper.res))
            # Monitor dump site throughput and queue status
            self.env.process(dump_site.monitor_status(env=self.env))
            # Monitor dumper throughput
            for dumper in dump_site.dumper_list:
                self.env.process(dumper.monitor_status(env=self.env))
        # Monitor mine-wide metrics
        self.env.process(self.monitor_status(env=self.env))
        # log in the truck as process
        for truck in self.trucks:
            self.env.process(truck.run())
        self.env.run(until=total_time)  # Run the simulation until the allotted time elapses

        # After the simulation ends, send the final observation and done flag
        ob = self.dispatcher.current_observation
        info = ob["info"]
        if reward_mode == "dense":
            reward = self.dispatcher._get_reward_dense(self)
        elif reward_mode == "sparse":
           reward = self.dispatcher._get_reward_dense(self)
        else:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        done = True
        self.env.done = done
        trucated = False
        out = {
            "ob": ob,
            "info": info,
            "reward": reward,
            "truncated": trucated,
            "done": done
        }
        self.mine_logger.info("simulation finished")
        self.summary()
        if ticks:
            self.dump_frames(total_time=total_time, rl=True)
        obs_queue.put(out, timeout=5)  # Push the observation into the queue

    def dump_frames(self, total_time, rl=False):
        """Use TickGenerator to capture simulation data and write it to disk."""
        # tick generator
        self.tick_generator = TickGenerator(mine=self, tick_num=total_time)
        assert self.tick_generator is not None, "tick_generator can not be None"

        print("dumping frames...")
        self.tick_generator.run()

        # Format current time as YYYY-MM-DD HH-MM-SS
        time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

        if rl:
            # Use the TickGenerator result path as the output directory
            result_path = self.tick_generator.result_path
            if not os.path.exists(result_path):
                os.makedirs(result_path)  # Create the directory if it does not exist

            # Locate existing episode files within the directory
            file_pattern = os.path.join(result_path, f'MINE-{self.name}-EP-*.json')
            files = glob.glob(file_pattern)

            # Parse episode numbers and timestamps from filenames
            episodes = []
            for file in files:
                try:
                    # Assume filename format 'MINE-{self.name}-EP-{episode}-TIME-{time_str}.json'
                    filename = os.path.basename(file)  # Consider filename only
                    parts = filename.split('-')
                    episode = int(parts[3])  # Extract episode number
                    file_time_str = '-'.join(parts[-5:]).split('.')[0]  # Extract timestamp
                    file_time = datetime.strptime(file_time_str, "%Y-%m-%d %H-%M-%S")
                    episodes.append((episode, file_time))
                except (ValueError, IndexError):
                    continue  # Skip files with unexpected naming patterns

            # Sort by timestamp to derive the next episode index
            current_episode = 1
            if episodes:
                episodes.sort(key=lambda x: x[1])
                current_episode = episodes[-1][0] + 1

            # Write RL simulation data file
            ticks = self.tick_generator.write_to_file(
                file_name=f'MINE-{self.name}-EP-{current_episode}-TIME-{time_str}.json')
        else:
            # Preserve the previous behavior for non-RL runs
            ticks = self.tick_generator.write_to_file(
                file_name=f'MINE-{self.name}-ALGO-{self.dispatcher.name}-TIME-{time_str}.json')

        return ticks

    @property
    def match_factor(self):
        # Compute the matching factor described in "Match factor for heterogeneous truck and loader fleets"
        shovels = [shovel for load_site in self.load_sites for shovel in load_site.shovel_list]
        trucks = self.trucks
        truck_cycle_time_avg = np.mean([truck.truck_cycle_time for truck in trucks])

        num_trucks = len(self.trucks)
        num_shovels = len(shovels)
        loading_time: np.array = np.zeros((num_trucks, num_shovels))
        for i, truck in enumerate(trucks):
            for j, shovel in enumerate(shovels):
                loading_time[i, j] = round((truck.truck_capacity / shovel.shovel_tons),
                                           1) * shovel.shovel_cycle_time  # in mins

        # Remove duplicates along rows and columns
        loading_time = np.unique(loading_time, axis=0)
        unique_loading_time = np.unique(loading_time, axis=1).astype(int)
        # Count shovel type occurrences for heterogeneous fleets
        shovel_type_count = dict()
        for i in range(unique_loading_time.shape[0]):  # truck type index
            int_data = np.array(loading_time[i, :]).astype(int)
            for value in set(int_data):
                shovel_type_count[f'{i}_{value}'] = list(int_data).count(
                    value)  # it means truck type i w.r.t shovel type num

        # unique_loading_time = np.ones_like(unique_loading_time) + unique_loading_time
    # Compute the LCM for each row
        lcm_load_time = np.lcm.reduce(unique_loading_time, axis=1)
        upside_down_sum = 0
        for i in range(unique_loading_time.shape[0]):
            for j in range(unique_loading_time.shape[1]):
                upside_down_sum += shovel_type_count[f'{i}_{unique_loading_time[i, j]}'] * (
                            lcm_load_time[i] / unique_loading_time[i, j])
        match_factor = (num_trucks * np.sum(lcm_load_time)) / (upside_down_sum * truck_cycle_time_avg)
        return match_factor