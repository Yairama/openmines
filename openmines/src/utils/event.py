from collections import defaultdict
from bisect import bisect_left


class Event:
    def __init__(self, time_stamp: float, event_type: str, desc: str, info: dict = None):
        self.time_stamp = time_stamp
        self.event_type = event_type
        self.desc = desc
        self.info = info

    def __str__(self):
        return f"Event(time_stamp={self.time_stamp},event_type={self.event_type},desc={self.desc},info={self.info})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if isinstance(other, Event):
            return self.time_stamp < other.time_stamp
        elif isinstance(other, (int, float)):
            return self.time_stamp < other
        return NotImplemented


class EventPool:
    """
    Container for truck operation events to simplify downstream analysis.
    """

    def __init__(self):
        self.event_set = dict()
        self.events_by_type = defaultdict(list)

    def add_event(self, event: Event):
        if event.time_stamp in self.event_set.keys():
            event.time_stamp = event.time_stamp + 0.0001
        self.event_set[event.time_stamp] = event

        t_list = self.events_by_type[event.event_type]
        idx = bisect_left(t_list, event)
        t_list.insert(idx, event)

    def get_even_by_type(self, name: str) -> list:
        """
        Retrieve events by their type name.
        """
        return self.events_by_type.get(name, [])

    def get_even_by_desc(self, name: str) -> list:
        """
        Retrieve events whose description contains the given text.
        """
        list_event = []
        for t in sorted(self.event_set.keys()):
            if name in self.event_set[t].desc:
                list_event.append(self.event_set[t])
        return list_event

    def get_event_by_time(self, time: float, mode="backward") -> list:
        """
        Return events ordered by time relative to the provided timestamp.

        If mode is "backward" (default) the result includes events at or before
        the timestamp. Any other mode returns events strictly after the timestamp.
        """
        if mode == "backward":
            list_event = []
            for t in sorted(self.event_set.keys()):
                if t <= time:
                    list_event.append(self.event_set[t])
            return list_event
        else:
            list_event = []
            for t in sorted(self.event_set.keys()):
                if t > time:
                    list_event.append(self.event_set[t])
            return list_event

    def get_event_by_time_range(self, start_time: float, end_time: float) -> list:
        """
        Return events whose timestamps fall within the inclusive range.
        """
        list_event = []
        for t in sorted(self.event_set.keys()):
            if start_time <= t <= end_time:
                list_event.append(self.event_set[t])
        return list_event

    def update_last_info(self, type: str, info: dict, strict: bool = True):
        """
        Update the info payload of the most recent matching event.

        When strict=True the most recent event must match the requested type.
        When strict=False the search walks backward until it finds a matching
        event and updates that entry.
        """
        if strict:
            assert self.event_set[list(self.event_set.keys())[-1]].event_type == type
            self.event_set[list(self.event_set.keys())[-1]].info = info
        else:
            for t in sorted(self.event_set.keys(), reverse=True):
                if type in self.event_set[t].event_type:
                    self.event_set[t].info = info
                    break

    def get_last_event(self, type: str, strict: bool = True):
        """
        Return the most recent event, honoring the strictness rules from above.

        Strict mode requires the latest event to match the requested type.
        Non-strict mode scans backward until it finds the first matching event.
        """
        if strict:
            assert self.event_set[list(self.event_set.keys())[-1]].event_type == type
            return self.event_set[list(self.event_set.keys())[-1]]
        else:
            for t in sorted(self.event_set.keys(), reverse=True):
                if type in self.event_set[t].event_type:
                    return self.event_set[t]
                    break

    def clear(self):
        self.event_set.clear()


class RandomEventPool:
    """
    Placeholder for random events captured during simulation.

    Intended both for retrospective analysis and as input to dispatch
    decisions while the simulation is running.
    """
    pass


if __name__ == "__main__":
    # test EventPool
    print("==============test EventPool================")
    pool = EventPool()
    pool.add_event(Event(3.0, 'test', 'desc'))
    pool.add_event(Event(1.0, 'test', 'desc'))
    pool.add_event(Event(1.0, 'test2', 'desc'))
    pool.add_event(Event(2.0, 'dasdasd', 'desc'))

    print(pool.get_even_by_type('test'))
    print(pool.get_even_by_desc('desc'))
    # clear
    pool.clear()
    # test update_last_info
    print("==============test update_last_info================")
    pool.add_event(Event(1.0, 'move', 'desc'))
    pool.add_event(Event(1.5, 'wait', 'desc', info={'load_time': 9527}))
    pool.add_event(Event(2.0, 'wait', 'desc', info={'wait_time': 0}))
    pool.add_event(Event(3.0, 'load', 'desc', info={'load_time': 98}))
    pool.update_last_info('wait', {'wait_time': 10}, strict=False)
    print(pool.get_even_by_type('wait'))

    # test get_last_event
    print("==============test get_last_event================")
    event1 = pool.get_last_event('wait', strict=False)
    print(f"BEFORE: {event1}")
    event1.info['wait_time'] = 100000
    print(f"AFTER: {pool.get_last_event('wait', strict=False)}")

    # test get_event_by_time
    print("==============test get_event_by_time================")
    for t in range(100):
        pool.add_event(Event(time_stamp=t, event_type='test', desc='desc'))
    print(pool.get_event_by_time(21))