from backend.fleet.vehicle import Vehicle


def test_vehicle_roundtrip_and_defaults():
    v = Vehicle("v000", 50, 7, 28800.0)
    assert v.status == "idle" and v.assigned_customers == [] and v.remaining_route == []
    v.remaining_route.append("c001")
    assert Vehicle.from_dict(v.to_dict()) == v
