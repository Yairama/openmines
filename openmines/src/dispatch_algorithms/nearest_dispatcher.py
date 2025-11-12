from __future__ import annotations

import time

import numpy as np

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite


class NearestDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "NearestDispatcher"

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        Dispatch the truck from the charging area to the nearest load site.
        :param truck: Truck instance
        :param mine: Mine instance
        :return: Index of the load site
        """
        charging_to_load = mine.road.charging_to_load
        # Find the index with the shortest distance
        min_index = charging_to_load.index(min(charging_to_load))
        return min_index

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        Send the truck to the closest dump site for unloading.
        :param truck: Truck instance
        :param mine: Mine instance
        :return: Index of the dump site
        """
        # Get the truck's current location
        current_location = truck.current_location
        if not isinstance(current_location, LoadSite):
            raise ValueError(f"Truck {truck.name} is not at a LoadSite: {current_location.name}")
        assert isinstance(current_location, LoadSite), f"current_location is not a LoadSite: {current_location.name}"
        # Obtain the index of that location
        cur_index = mine.load_sites.index(current_location)
        # Fetch distances to all dump sites
        cur_to_dump = mine.road.l2d_road_matrix[cur_index,:]
        # Identify the index with the shortest distance
        min_index = cur_to_dump.argmin()
        return min_index

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        Route the truck to the nearest load site to pick up material.
        :param truck: Truck instance
        :param mine: Mine instance
        :return: Index of the load site
        """
        # Get the truck's current location
        current_location = truck.current_location
        if not isinstance(current_location, DumpSite):
            raise ValueError(f"Truck {truck.name} is not at a DumpSite: {current_location.name}")
        assert isinstance(current_location, DumpSite), f"current_location is not a DumpSite: {current_location.name}"
        # Obtain the index of that location
        cur_index = mine.dump_sites.index(current_location)
        # Fetch distances to all load sites
        cur_to_load = mine.road.d2l_road_matrix[:,cur_index]
        # Identify the index with the shortest distance
        min_index = cur_to_load.argmin()
        return min_index

if __name__ == "__main__":
    dispatcher = NearestDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))

    print(dispatcher.total_order_count,dispatcher.init_order_count,dispatcher.init_order_time,dispatcher.total_order_time)
