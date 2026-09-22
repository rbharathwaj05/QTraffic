from backend.fleet.customer import Customer


def test_customer_roundtrip():
    c = Customer("c000", 13.0, 80.0, 42, 5, 300, 3600.0, 7200.0)
    assert Customer.from_dict(c.to_dict()) == c
    assert c.to_dict()["node_id"] == 42
