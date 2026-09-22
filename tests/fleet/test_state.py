import numpy as np

from backend.fleet.state import FleetState
from backend.optimization.encoding import FleetRoute


def fleet():
    return FleetRoute([np.array([0, 1, 2, 0]), np.array([0, 3, 0])], np.array([0, 0, 1]))


def test_deploy_starts_every_vehicle_at_the_depot_with_nothing_served():
    s = FleetState.from_fleet_route(fleet(), t_sim=0.0)
    assert s.n_vehicles == 2
    assert s.position.tolist() == [0, 0] and s.served == set()
    assert s.last_reopt.tolist() == [0.0, 0.0]
    assert s.remaining(0).tolist() == [0, 1, 2, 0]


def test_advance_marks_customers_served_and_stops_at_the_depot():
    s = FleetState.from_fleet_route(fleet())
    s.advance(0)
    assert s.position[0] == 1 and s.served == {1}
    s.advance(0, stops=10)  # cannot run past the end of the route
    assert s.position[0] == 3 and s.served == {1, 2}
    assert s.remaining_legs(0) == [] and s.remaining_customers(0).tolist() == []


def test_remaining_view_shrinks_as_the_vehicle_drives():
    s = FleetState.from_fleet_route(fleet())
    assert s.remaining_customers(0).tolist() == [1, 2]
    s.advance(0)
    assert s.remaining_customers(0).tolist() == [2]
    assert s.remaining_legs(0) == [(1, 2), (2, 0)]


def test_apply_replan_keeps_the_frozen_prefix_and_stamps_the_cooldown():
    """[SPEC 9.3 point 1 / v4 47] splice, do not rewrite; the vehicle stays where it is."""
    s = FleetState.from_fleet_route(fleet(), t_sim=0.0)
    s.advance(0)  # at customer 1
    s.apply_replan(0, np.array([1, 2, 0]), t_sim=400.0)
    assert s.routes[0].tolist() == [0, 1, 2, 0]
    assert s.position[0] == 1  # same stop, new plan ahead of it
    assert s.last_reopt[0] == 400.0 and s.last_reopt[1] == 0.0


def test_apply_replan_can_reorder_the_untravelled_suffix():
    s = FleetState.from_fleet_route(fleet())
    s.advance(0)
    s.apply_replan(0, np.array([1, 3, 2, 0]), t_sim=10.0)
    assert s.routes[0].tolist() == [0, 1, 3, 2, 0]
    assert s.remaining_customers(0).tolist() == [3, 2]


def test_record_replan_stamps_only_the_listed_vehicles():
    s = FleetState.from_fleet_route(fleet(), t_sim=-np.inf)
    s.record_replan([1], 250.0)
    assert s.last_reopt[1] == 250.0 and s.last_reopt[0] == -np.inf


def test_round_trip_back_to_a_fleet_route_is_a_copy():
    s = FleetState.from_fleet_route(fleet())
    out = s.to_fleet_route()
    out.routes[0][1] = 99
    assert s.routes[0][1] == 1  # the state was not aliased into the report artifact
