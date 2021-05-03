lint:
	pre-commit run --all-files

test:
	pytest -v tf_autoaugment/tests
