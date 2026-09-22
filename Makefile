.PHONY: check lint test fmt osrm-build
check: lint test
lint:
	ruff check .
	black --check .
fmt:
	ruff check --fix .
	black .
test:
	pytest -q
# one-time OSRM data build: data/city/city.osm.pbf -> city.osrm (see backend/road/osrm_client.py)
osrm-build:
	docker compose run --rm osrm osrm-extract -p /opt/car.lua /data/city.osm.pbf
	docker compose run --rm osrm osrm-partition /data/city.osrm
	docker compose run --rm osrm osrm-customize /data/city.osrm
