import numpy as np

def preprocess_observation(observation, max_sim_time):
    """Feature preprocessing logic for RL observations."""
    """
    0. Order information
    """
    # 1. Order type and timing information
    event_name = observation['event_name']
    if event_name == "init":
        event_type = [1, 0, 0]
        action_space_n = observation['info']['load_num']
    elif event_name == "haul":
        event_type = [0, 1, 0]
        action_space_n = observation['info']['unload_num']
    else:
        event_type = [0, 0, 1]
        action_space_n = observation['info']['load_num']
    # 2. Absolute and relative positions of the current order time
    time_delta = float(observation['info']['delta_time'])  # Time elapsed since the last dispatch
    time_now = float(observation['info']['time']) / max_sim_time  # Current time (normalized)
    time_left = 1 - time_now  # Remaining normalized time
    order_state = np.array([event_type[0], event_type[1], event_type[2], time_delta, time_now, time_left])

    """
    1. Vehicle-specific information
    """
    # Total number of trucks in the mine (used for normalization)
    truck_num = observation['mine_status']['truck_count']
    # 4. One-hot encoding of the truck's current location
    truck_location_onehot = np.array(observation["the_truck_status"]["truck_location_onehot"])
    # Truck load and cycle time features (log-normalized)
    truck_features = np.array([
        np.log(observation['the_truck_status']['truck_load'] + 1),
        np.log(observation['the_truck_status']['truck_cycle_time'] + 1),
    ])
    truck_self_state = np.concatenate([truck_location_onehot, truck_features])

    """
    2. Road-related information
    """
    # Expected travel times
    travel_time = np.array(observation['cur_road_status']['distances']) * 60 / 25
    # Number of trucks on each road segment
    truck_counts = np.array(observation['cur_road_status']['truck_counts']) / (truck_num + 1e-8)
    # Road distance features
    road_dist = np.array(observation['cur_road_status']['oh_distances'])
    # Road congestion indicators
    road_jam = np.array(observation['cur_road_status']['oh_truck_jam_count'])

    road_states = np.concatenate([travel_time, truck_counts, road_dist, road_jam])

    """
    3. Target-site information
    """
    # Expected waiting times
    est_wait = np.log(observation['target_status']['single_est_wait'] + 1)  # Includes on-road trucks and queued trucks for the target load site
    tar_wait_time = np.log(np.array(observation['target_status']['est_wait']) + 1)  # Excludes trucks currently on the road
    # Normalized queue lengths
    queue_lens = np.array(observation['target_status']['queue_lengths']) / (truck_num + 1e-8)
    # Load capacities
    tar_capa = np.log(np.array(observation['target_status']['capacities']) + 1)
    # Current productivity ratio for each target (maintenance may reduce capacity)
    ability_ratio = np.array(observation['target_status']['service_ratio'])
    # Cumulative produced tonnage (log-normalized)
    produced_tons = np.log(np.array(observation['target_status']['produced_tons']) + 1)

    tar_state = np.concatenate([est_wait, tar_wait_time, queue_lens, tar_capa, ability_ratio, produced_tons])

    state = np.concatenate([order_state, truck_self_state, road_states, tar_state])
    assert not np.isnan(state).any(), f"NaN detected in state: {state}"
    assert not np.isnan(time_delta), f"NaN detected in time_delta: {time_delta}"
    assert not np.isnan(time_now), f"NaN detected in time_now: {time_now}"

    return state.astype(np.float32) 