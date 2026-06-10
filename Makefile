.PHONY: install setup demo kaggle features train pipeline experiment test serve

install:
	python3 -m pip install -e ".[dev]"

setup: install
	python3 scripts/make_dataset.py

demo:
	python3 main.py demo-data

kaggle:
	python3 main.py prepare-kaggle

train:
	python3 main.py train

features:
	python3 scripts/build_features.py

pipeline:
	python3 main.py pipeline

experiment:
	python3 main.py experiment

test:
	python3 -m pytest -q

serve:
	python3 main.py serve --reload
