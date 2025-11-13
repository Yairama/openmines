import random

import numpy as np
import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from simpy.resources.resource import Request

from openmines.src.charging_site import ChargingSite
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.utils.event import Event, EventPool

TRUCK_DEFAULT_SPEED = 25 # km/h
"""Post-simulation truck trajectory export format.

Each simulation frame is converted into a DataFrame row with the following
conventions:
    start/dest label:
        -1 -> charging site
        0-N -> load sites
        N+1 to N+M -> dump sites
    state:
        0 -> empty
        1 -> waiting to load
        2 -> loading
        3 -> fully loaded
        4 -> waiting to unload
        5 -> unloading
    position_ratio:
        0-1 indicates progress along the current leg (useful during states 0-3)
    position_x, position_y:
        Coordinates for load/dump/shovel locations supplied via configuration;
        the simulation precomputes the 2D positions here.
    time:
        Simulation timestamp

TODO: handle overlapping vehicles in visualizations (potentially integrate
SUMO or similar tooling).
"""

class LoadRequest(Request):
    """
    SimPy request wrapper used when a truck reserves shovel capacity.
    """
    def __init__(self, resource, truck, load_site):
        super().__init__(resource)
        self.truck = truck
        self.load_site = load_site

class DumpRequest(Request):
    def __init__(self, resource, truck, dump_site):
        super().__init__(resource)
        self.truck = truck
        self.dump_site = dump_site

class Truck:
    def __init__(self, name:str, truck_capacity:float, truck_speed:float=TRUCK_DEFAULT_SPEED):
        self.name = name
        self.truck_capacity = truck_capacity  # truck capacity in tons
        self.truck_speed = truck_speed  # truck speed in km/h
        self.current_location = None
        self.target_location = None
        self.journey_start_time = 0
        self.journey_coverage = None
        self.pre_jam_time = 0
        self.last_breakdown_time = 0
        self.event_pool = EventPool()
        # the probalistics of the truck
        self.expected_working_time_without_breakdown = 60*6  # in minutes
        self.expected_working_time_until_unrepairable = 60*24*7*7  # in minutes
        self.repair_avg_time = 10
        self.repair_std_time = 3
        self.repair_time = 0
        # truck status
        self.status = "idle"
        self.truck_load = 0  # in tons, the current load of the truck
        self.service_count = 0  # the number of times the truck has been dumped
        self.total_load_count = 0  # the total load count of the truck
        self.truck_cycle_time = 0
        self.first_order_time = 0
        # RL
        self.current_decision_event = None

    def get_location_onehot(self):
        """Return a one-hot encoding for the truck's current location."""
        charge_pos, load_pos, dump_pos = [0],[0]*len(self.mine.load_sites),[0]*len(self.mine.dump_sites)
        if isinstance(self.current_location, DumpSite):
            event_name = "unhaul"  # at dump site
            for i, dump_site in enumerate(self.mine.dump_sites):
                if dump_site.name == self.current_location.name:
                    dump_pos[i] = 1
                    return charge_pos + load_pos + dump_pos
        elif isinstance(self.current_location, ChargingSite):
            event_name = "init" # at charge site
            charge_pos[0] = 1
            return  charge_pos + load_pos + dump_pos
        else:
            event_name = "haul"  # at load site
            for i, load_site in enumerate(self.mine.load_sites):
                if load_site.name == self.current_location.name:
                    load_pos[i] = 1
                    return charge_pos + load_pos + dump_pos
        return charge_pos + load_pos + dump_pos

    def set_env(self, mine:"Mine"):
        self.mine = mine
        self.env = mine.env
        self.dispatcher = mine.dispatcher

    def move(self, target_location, distance:float, manual_speed:float=None):
        """
        Move the truck toward the target location while modeling loaded vs. empty travel.

        Args:
            target_location: Destination site object.
            distance: Road distance to travel (kilometers).
            manual_speed: Optional override speed in km/h.
        """
        assert target_location is not None, "target_location can not be None"
        assert target_location is not self.current_location, "target_location can not be the same as current_location"
        assert distance >= 0, "distance can not be negative"
        assert self.truck_speed > 0, "truck_speed can not be negative"
        # Record details for the current trip
        self.target_location = target_location
        self.journey_start_time = self.env.now
        if manual_speed is not None:
            assert manual_speed > 0, "manual_speed can not be negative"
            duration = (distance / manual_speed)*60  # in minutes
        else:
            duration = (distance / self.truck_speed)*60
        self.truck_speed = manual_speed if manual_speed is not None else self.truck_speed

        """
        1. Simulate stochastic repair events.
        """
        # Check whether the truck breaks down and obtain the repair duration
        repair_time = self.check_vehicle_availability()
        if repair_time:
            # If a breakdown occurs, record the event and perform the repair
            breakdown_event = Event(self.last_breakdown_time, "TruckEvent:breakdown", f'Time:<{self.last_breakdown_time}> Truck:[{self.name}] breakdown for {repair_time} minutes',
                      info={"name": self.name, "status": "breakdown",
                            "repair_time": repair_time,
                            "start_location": self.current_location.name,
                            "target_location": self.target_location.name,
                            "start_time": self.last_breakdown_time, "end_time": self.last_breakdown_time + repair_time})
            self.event_pool.add_event(breakdown_event)
            self.mine.random_event_pool.add_event(breakdown_event)
            self.logger.info(f'Time:<{self.last_breakdown_time}> Truck:[{self.name}] breakdown for {repair_time} minutes at {self.last_breakdown_time}')
            # Pause the trip to simulate the repair period
            self.status = "repairing"
            yield self.env.timeout(repair_time)
            self.repair_time = 0
            self.status = "moving"
        """
        2. Simulate stochastic traffic jam events during transit.
        """
        # Analyze current road conditions and sample a delay to mimic congestion
        # Retrieve existing jam events
        jam_events = self.mine.random_event_pool.get_even_by_type("RoadEvent:jam")
        pre_jam_time = 0
        pre_jam_count = 0
        for jam_event in jam_events:
                        """
                        info={"name": self.name, "status": "jam", "speed": 0,
                                                                                                    "start_location": self.current_location.name,
                                                                                                        "end_location": self.target_location.name,
                                                                                                        "jam_position": jam_position,
                                                                                                    "start_time": self.env.now, "est_end_time":
                        """
            if jam_event.info["start_location"] == self.current_location.name and jam_event.info["end_location"] == self.target_location.name \
                    and jam_event.info["start_time"] <= self.env.now and jam_event.info["est_end_time"] >= self.env.now:
                jam_time = jam_event.info["est_end_time"] - self.env.now  # Remaining jam duration at departure
                jam_position = jam_event.info["jam_position"]
                time_to_jam = jam_position * duration  # Time until the truck reaches the jam segment
                pre_jam_time += max(0, jam_time - time_to_jam)
                pre_jam_count += 1
                self.pre_jam_time = pre_jam_time
        if pre_jam_count > 0:
            self.logger.info(f"Time:<{self.env.now}> Truck:[{self.name}] is facing {pre_jam_time} mins delay caused by {pre_jam_count} pre-existing jam events on Road from {self.current_location.name} to {self.target_location.name}")

        """
        3. Simulate total vehicle failure (truck removed from service).
        """
        if repair_time is None:
            self.logger.info(f"Time:<{self.last_breakdown_time}> Truck:[{self.name}] is broken down and beyond repair at {self.current_location.name} to "
                             f"{self.target_location.name}")
            unrepairable_event = Event(self.env.now, "TruckEvent:unrepairable", f'Time:<{self.env.now}> Truck:[{self.name}] is broken down and beyond repair'
                                                                     f' at {self.current_location.name} to '
                                                                        f'{self.target_location.name}',
                                        info={"name": self.name, "status": "unrepairable","time": self.env.now})
            self.event_pool.add_event(unrepairable_event)
            self.mine.random_event_pool.add_event(unrepairable_event)
            self.status = "unrepairable"
            return
        """
        4. Simulate travel toward the destination.
        """
        # Determine the type of destination
        if isinstance(self.current_location, DumpSite) and isinstance(target_location, LoadSite):
            event_name = "unhaul"
        elif isinstance(self.current_location, ChargingSite):
            event_name = "init"
        else:
            event_name = "haul"

        self.event_pool.add_event(Event(self.env.now, event_name, f'Truck:[{self.name}] moves at {target_location.name}',
                                        info={"name": self.name, "status": event_name, "speed": manual_speed if manual_speed is not None else self.truck_speed,
                                              "start_time": self.env.now, "est_end_time": self.env.now + duration + pre_jam_time, "end_time": None,
                                              "start_location": self.current_location.name,
                                              "target_location": target_location.name,
                                              "distance": distance, "duration": None}))
        self.status = "moving"
        yield self.env.timeout(duration + pre_jam_time)
        # Complete event metadata after arrival
        last_move_event = self.event_pool.get_last_event(type=event_name, strict=True)
        last_move_event.info["end_time"] = self.env.now
        last_move_event.info["duration"] = self.env.now - last_move_event.info["start_time"]
        # after arrival set current location
        assert type(self.current_location) != type(target_location), f"current_site and target_site should not be the same type of site "
        self.current_location = target_location
        self.pre_jam_time = 0

    def load(self, shovel:Shovel):
        shovel_tons = shovel.shovel_tons
        shovel_cycle_time = shovel.shovel_cycle_time
        load_time = (self.truck_capacity/shovel_tons) * shovel_cycle_time
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Start loading at shovel {shovel.name}, load time is {load_time}')
        self.status = "loading"
        shovel.last_service_time = self.env.now  # Track last service start for wait estimation
        shovel.load_site.update_service_time()
        yield self.env.timeout(load_time)
        # Randomize the actual load by ±10%
        self.truck_load = self.truck_capacity*(1+np.random.uniform(-0.1, 0.1))
        shovel.produced_tons += self.truck_load
        shovel.service_count += 1
        shovel.last_service_done_time = self.env.now  # Track last service completion time
        shovel.load_site.update_service_time()
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Finish loading at shovel {shovel.name}')

    def unload(self, dumper:Dumper):
        unload_time:float = dumper.dump_time
        self.status = "unloading"
        dumper.last_service_time = self.env.now  # Track last dump start for wait estimation
        dumper.dump_site.update_service_time()
        yield self.env.timeout(unload_time)
        dumper.dumper_tons += self.truck_load
        self.total_load_count += self.truck_load
        self.service_count += 1
        self.truck_cycle_time = (self.env.now - self.first_order_time) / self.service_count
        self.truck_load = 0
        dumper.service_count += 1
        dumper.last_service_done_time = self.env.now  # Track last dump completion time
        dumper.dump_site.update_service_time()
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Finish unloading at dumper {dumper.name}, dumper tons is {dumper.dumper_tons}')
        # self.event_pool.add_event(Event(self.env.now, "unload", f'Truck:[{self.name}] Finish unloading at dumper {dumper.name}',
        #                           info={"name": self.name, "status": "unloading",
        #                                 "start_time": self.env.now-unload_time, "end_time": self.env.now,
        #                                 "dumper": dumper.name, "unload_duration": unload_time}))
        # WARN: "unload" duplicates data captured by "get dumper"

    def check_queue_position(self, shovel, request):
        """
        Inspect the shovel queue and return the truck's position.

        Args:
            shovel: Shovel whose queue should be inspected.
            request: The truck's resource request object.

        Returns:
            Zero-based index of the truck in the queue.
        """
        try:
            if len(shovel.res.queue) == 0:
                return 0
            elif shovel.res.queue.index(request) == 0:
                return 0
            else:
                return shovel.res.queue.index(request)
        except ValueError:
            return 0  # Request no longer in queue; treat as front of line

    def wait_for_decision(self):
        # Create an event the RL agent can trigger later
        self.current_decision_event = self.env.event()
        yield self.current_decision_event  # Wait for the RL environment decision
        return self.current_decision_event.value  # Return the chosen action

    def run(self, is_rl_training=False):
        """
        Core coroutine controlling truck behavior through the dispatch cycle.

        Flow:
            1. Start at the charging site and request an init order to a load site.
            2. Load at the assigned shovel.
            3. Request a haul order and travel to a dump site.
            4. Unload at the assigned dumper.
            5. Request a back order and return to a load site.

        TODO:
            1. Model arrival times with a normal distribution.
            2. Differentiate electric vs. diesel trucks, including energy usage.
            3. Extend the catalogue of random events.
            4. Produce additional monitoring visualizations.
        """
        # Configure logger
        self.logger = self.mine.global_logger.get_logger("Truck")

        # Shift begins: truck departs from charging area toward load site
        self.current_location = self.mine.charging_site
        self.status = "waiting for init order"
        if is_rl_training:
            # Wait for the RL agent to provide a decision
            dest_load_index: int = yield self.env.process(self.wait_for_decision())
        else:
            dest_load_index: int = self.dispatcher.give_init_order(truck=self, mine=self.mine)  # TODO: allow speed planning
        self.status = "moving"
        self.first_order_time = self.env.now
        self.target_location = self.mine.load_sites[dest_load_index]
        self.mine.update_road_status()  # Manually update road status before monitors begin

        move_distance:float = self.mine.road.charging_to_load[dest_load_index]
        load_site: LoadSite = self.mine.load_sites[dest_load_index]
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Activated at {self.env.now}, Target load site is ORDER({dest_load_index}):{load_site.name}, move distance is {move_distance}')
        self.event_pool.add_event(Event(self.env.now, "INIT ORDER", f'Truck:[{self.name}] Activated at {self.env.now}, Target load site is ORDER({dest_load_index}):{load_site.name}, move distance is {move_distance}',
                                        info={"name": self.name, "status": "INIT ORDER",
                                              "start_time": self.env.now, "est_end_time": self.env.now+move_distance/self.truck_speed,
                                              "start_location": self.current_location.name,
                                              "target_location": load_site.name,
                                              "distance": move_distance, "order_index": dest_load_index}))
        yield self.env.process(self.move(target_location=load_site, distance=move_distance))  # Travel time
        # Exit early if the truck has become unrepairable
        if self.status == "unrepairable":
            self.logger.info(f"Truck {self.name} is beyond repair and will no longer participate in operations.")
            return

        while True:
            # Arrived at load site: request shovel resource and begin loading
            self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Arrived at {self.mine.load_sites[dest_load_index].name} at {self.env.now}')
            load_site:LoadSite = self.mine.load_sites[dest_load_index]
            shovel = load_site.get_available_shovel()

            with LoadRequest(shovel.res, self, load_site) as req:
                # Before resource acquisition
                truck_queue_index = self.check_queue_position(shovel, req)
                self.event_pool.add_event(Event(self.env.now, "wait shovel", f'Truck:[{self.name}] Wait shovel {shovel.name}',
                                                info={"name": self.name, "status": "waiting for shovel",
                                                      "queue_index": truck_queue_index,
                                                      "start_time": self.env.now, "end_time": None,
                                                      "shovel": shovel.name, "wait_duration": None}))
                self.status = "waiting for shovel"
                # ...
                yield req  # Request shovel resource
                # Resource granted
                # Update the prior wait event with completion data
                last_wait_event = self.event_pool.get_last_event(type="wait shovel", strict=True)
                last_wait_event.info["end_time"] = self.env.now
                last_wait_event.info["wait_duration"] = self.env.now - last_wait_event.info["start_time"]
                # Record the load event
                self.event_pool.add_event(Event(self.env.now, "get shovel", f'Truck:[{self.name}] Get shovel {shovel.name}',
                                                info={"name": self.name, "status": "loading on shovel",
                                                      "start_time": self.env.now, "end_time": None,
                                                      "shovel": shovel.name, "load_duration": None}))
                yield self.env.process(self.load(shovel))  # Loading duration depends on both shovel and truck
                # Loading complete: update event metadata
                last_load_event = self.event_pool.get_last_event(type="get shovel", strict=True)
                last_load_event.info["end_time"] = self.env.now
                last_load_event.info["load_duration"] = self.env.now - last_load_event.info["start_time"]
                # **Shovel maintenance logic**
                # Failure events sampled from an exponential distribution (lambda=1/480)
                # Maintenance duration sampled from Normal(mu=45, sigma=5)
                time_to_next_maintenance = np.random.exponential(scale=60*24)  # Average 480 minutes
                if self.env.now >= shovel.last_breakdown_time + time_to_next_maintenance:
                    # Trigger maintenance
                    shovel.repair = True
                    maintenance_duration = max(np.random.normal(45, 5), 0)  # Ensure non-negative duration
                    # Log maintenance start
                    shovel.event_pool.add_event(Event(self.env.now, "Shovel Maintenance Start",
                                                    f'Shovel {shovel.name} under maintenance for {maintenance_duration:.2f} minutes',
                                                    info={"shovel": shovel.name, "status": "maintenance",
                                                          "start_time": self.env.now, "end_time": self.env.now + maintenance_duration}))
                    self.logger.info(f'Time:<{self.env.now}> Shovel {shovel.name} under maintenance for {maintenance_duration:.2f} minutes')
                    # Perform maintenance while holding the resource
                    yield self.env.timeout(maintenance_duration)
                    # Log maintenance completion
                    shovel.repair = True
                    shovel.event_pool.add_event(Event(self.env.now, "Shovel Maintenance Complete",
                                                    f'Shovel {shovel.name} maintenance completed',
                                                    info={"shovel": shovel.name, "status": "available",
                                                          "end_time": self.env.now}))
                    self.logger.info(f'Time:<{self.env.now}> Shovel {shovel.name} maintenance completed')
                    # Update last maintenance timestamp
                    shovel.last_breakdown_time = self.env.now
                # **End shovel maintenance logic**

            # After loading, request a dump site and travel there
            self.status = "waiting for haul order"
            self.mine.update_road_status()  # Manually refresh road status
            if is_rl_training:
                # Wait for the RL agent to provide a decision
                dest_unload_index: int = yield self.env.process(self.wait_for_decision())
            else:
                dest_unload_index: int = self.dispatcher.give_haul_order(truck=self, mine=self.mine)
            dest_unload_site: DumpSite = self.mine.dump_sites[dest_unload_index]
            move_distance: float = self.mine.road.get_distance(truck=self, target_site=dest_unload_site)
            self.target_location = dest_unload_site

            self.logger.debug(f"Time:<{self.env.now}> Truck:[{self.name}] Start moving to ORDER({dest_unload_index}): {dest_unload_site.name}, move distance is {move_distance}, speed: {self.truck_speed}")
            self.event_pool.add_event(Event(self.env.now, "ORDER", f'Truck:[{self.name}] Start moving to ORDER({dest_unload_index}): {dest_unload_site.name}, move distance is {move_distance}, speed: {self.truck_speed}',
                                            info={"name": self.name, "status": "ORDER",
                                                  "start_time": self.env.now, "est_end_time": self.env.now+(move_distance/self.truck_speed)*60,
                                                  "start_location": self.current_location.name,
                                                  "target_location": dest_unload_site.name, "speed": self.truck_speed,
                                                  "distance": move_distance, "order_index": dest_unload_index}))

            yield self.env.process(self.move(target_location=dest_unload_site, distance=move_distance))  # Travel time

            # Exit loop if the truck has become unrepairable
            if self.status == "unrepairable":
                self.logger.info(f"Truck {self.name} is beyond repair and will no longer participate in operations.")
                return
            # Arrive at dump site, request dumper, and unload
            self.logger.debug(f'Time:<{self.env.now}> Truck:[{self.name}] Arrived at {dest_unload_site.name} at {self.env.now}')
            dumper:Dumper = dest_unload_site.get_available_dumper()
            with DumpRequest(dumper.res, self, dest_unload_site) as req:
                # Before resource acquisition
                # ...
                self.event_pool.add_event(Event(self.env.now, "wait dumper", f'Truck:[{self.name}] Wait dumper {dumper.name}',
                                                info={"name": self.name, "status": "waiting for dumper",
                                                      "start_time": self.env.now, "end_time": None,
                                                      "dumper": dumper.name, "wait_duration": None}))
                self.status = "waiting for dumper"
                yield req  # Request dumper resource
                # Resource granted
                # Update the prior wait event with completion data
                last_wait_event = self.event_pool.get_last_event(type="wait dumper", strict=True)
                last_wait_event.info["end_time"] = self.env.now
                last_wait_event.info["wait_duration"] = self.env.now - last_wait_event.info["start_time"]
                self.event_pool.add_event(Event(self.env.now, "get dumper", f'Truck:[{self.name}] Get dumper {dumper.name}',
                                                info={"name": self.name, "status": "unloading on dumper",
                                                      "start_time": self.env.now, "end_time": None,
                                                      "dumper": dumper.name, "unload_duration": None}))
                yield self.env.process(self.unload(dumper))
                # Unloading complete: update event metadata
                last_unload_event = self.event_pool.get_last_event(type="get dumper", strict=True)
                last_unload_event.info["end_time"] = self.env.now
                last_unload_event.info["unload_duration"] = self.env.now - last_unload_event.info["start_time"]

            # After unloading, request a load site and travel back
            self.status = "waiting for back order"
            if is_rl_training:
                # Wait for the RL agent to provide a decision
                dest_load_index: int = yield self.env.process(self.wait_for_decision())
            else:
                dest_load_index: int = self.dispatcher.give_back_order(truck=self, mine=self.mine)
            dest_load_site: LoadSite = self.mine.load_sites[dest_load_index]
            move_distance: float = self.mine.road.get_distance(truck=self, target_site=dest_load_site)
            self.target_location = dest_load_site
            self.mine.update_road_status()  # Manually refresh road status
            self.logger.debug(f"Time:<{self.env.now}> Truck:[{self.name}] Start moving to ORDER({dest_load_index}):{dest_load_site.name}, move distance is {move_distance}, speed: {self.truck_speed}")
            self.event_pool.add_event(Event(self.env.now, "ORDER", f'Truck:[{self.name}] Start moving to ORDER({dest_load_index}):{dest_load_site.name}, move distance is {move_distance}, speed: {self.truck_speed}',
                                            info={"name": self.name, "status": "ORDER",
                                                  "start_time": self.env.now, "est_end_time": self.env.now+move_distance/self.truck_speed,
                                                  "start_location": self.current_location.name,
                                                  "target_location": dest_load_site.name, "speed": self.truck_speed,
                                                  "distance": move_distance, "order_index": dest_load_index}))
            yield self.env.process(self.move(target_location=dest_load_site, distance=move_distance))  # Travel time
            # Exit loop if the truck has become unrepairable
            if self.status == "unrepairable":
                self.logger.info(f"Truck {self.name} is beyond repair and will no longer participate in operations.")
                return

    def charge(self, duration):
        """
        Handle charging for electric trucks.

        Diesel trucks ignore this routine.

        TODO: Model fuel consumption, battery usage, and charging time to feed
        downstream analytics.
        """
        self.logger.info(f'{self.name} Start charging at {self.env.now}')
        yield self.env.timeout(duration)

    def get_wait_time(self):
        """
        Compute the average waiting time per cycle using unload events.
        """
        # TODO: include trucks that never acquired resources so wait time reflects the full queue
        wait_shovel_events = self.event_pool.get_even_by_type("wait shovel")
        end_wait_shovel_events = self.event_pool.get_even_by_type("get shovel")
        self.event_pool.add_event(Event(self.mine.total_time, "end", f'Truck:[{self.name}] End'))
        end_event = self.event_pool.get_even_by_type("end")

        wait_shovel_event_count = len(wait_shovel_events)
        end_wait_shovel_event_count = len(end_wait_shovel_events)

        if wait_shovel_event_count > end_wait_shovel_event_count:
            # Simulation ended while still waiting; append the terminal event as completion
            end_wait_shovel_events.append(end_event[0])  # End event not recorded previously

        wait_shovel_event_time = sum([event.time_stamp for event in wait_shovel_events])
        end_wait_shovel_event_time = sum([event.time_stamp for event in end_wait_shovel_events])
        wait_time = (end_wait_shovel_event_time - wait_shovel_event_time) / wait_shovel_event_count if wait_shovel_event_count else 0
        if wait_time == float('inf'):
            wait_time = 0
        return wait_time

    def get_route_coverage(self,distance)->float:
        """
        Approximate the route completion ratio for the current cycle.

        Returns a value between 0 (just departed) and 1 (arrived). Traffic and
        other stochastic events are handled via separate adjustments.

        TODO: incorporate dynamic traffic conditions and breakdowns for a
        refined coverage estimate.
        """
        assert distance > 0, "distance must be greater than 0"
        if self.journey_start_time is None:
            print(1)
        assert self.journey_start_time is not None, "journey_start_time must be not None, is the truck journey started?"

        # Current simulation time
        current_time = self.env.now
        # Truck speed
        speed = self.truck_speed
        # Expected travel duration in minutes
        total_travel_time = (distance/speed)*60 + self.pre_jam_time + self.repair_time
        # Coverage ratio
        coverage = (current_time - self.journey_start_time)/total_travel_time
        # TODO: trigger road maintenance events via the monitor to improve distance estimates
        if coverage > 1:
            coverage = 1
        return coverage

    def sample_breakdown(self):
        """
        Sample the next potential breakdown time from an exponential distribution.
        """
        # Sample working time before the next breakdown
        time_to_breakdown = np.random.exponential(self.expected_working_time_without_breakdown)
        # Determine whether the breakdown occurs within the current step
        if self.env.now >= self.last_breakdown_time + time_to_breakdown:
            # Sample repair time via Normal(mu=10, sigma=3)
            repair_time = np.random.normal(self.repair_avg_time, self.repair_std_time)
            repair_time = max(repair_time, 0)  # Ensure positive repair duration
            # Update last breakdown timestamp
            self.last_breakdown_time = self.env.now  # self.last_breakdown_time + time_to_breakdown
            self.repair_time = repair_time

    def check_vehicle_availability(self):
        """
        Model truck availability with an exponential distribution.

        Returns:
            Repair duration if a breakdown occurs, otherwise zero. Returns
            None when the truck becomes permanently unavailable.
        """
        # Sample the time until the truck becomes unrepairable; return None when exceeded
        if self.env.now >= random.expovariate(1.0 / self.expected_working_time_until_unrepairable):
            return None
        return self.repair_time  # Return repair duration when a fault has been sampled
