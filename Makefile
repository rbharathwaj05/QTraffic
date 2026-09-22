.PHONY: check lint test fmt
check: lint test
lint:
	ruff check .
	black --check .
fmt:
	ruff check --fix .
	black .
test:
	pytest -q
