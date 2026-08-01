.PHONY: install setup data train evaluate pipeline app test clean

install:
	pip install -r requirements.txt

setup:
	python setup.py

data:
	python main.py prepare-data

train:
	python main.py train

evaluate:
	python main.py evaluate

pipeline:
	python main.py pipeline

app:
	streamlit run main.py

test:
	pytest -v

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

