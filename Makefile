lint:
	pre-commit run --all-files

test:
	pytest --disable-pytest-warnings -v tf_autoaugment/tests
