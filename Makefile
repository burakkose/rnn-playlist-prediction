install-deps:
	pip install -r requirements.txt

test:
	nosetests tests
