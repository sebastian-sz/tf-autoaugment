lint:
	pre-commit run --all-files

test:
	pytest --disable-pytest-warnings -v tf_autoaugment/tests

clean:
	find . -name "*.py[co]" -o -name __pycache__ -exec rm -rf {} +
