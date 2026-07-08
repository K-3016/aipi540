.PHONY: data train train-transformer evaluate experiment app clean

data:
	PYTHONPATH=src python scripts/make_dataset.py

train:
	PYTHONPATH=src python scripts/train_all_models.py

train-transformer:
	PYTHONPATH=src python scripts/train_all_models.py --include-transformer

evaluate:
	MPLCONFIGDIR=/private/tmp/matplotlib PYTHONPATH=src python -m campus_triage.evaluate

experiment:
	MPLCONFIGDIR=/private/tmp/matplotlib PYTHONPATH=src python scripts/run_experiment.py

app:
	PYTHONPATH=src streamlit run main.py

clean:
	rm -rf __pycache__ src/campus_triage/__pycache__ scripts/__pycache__ .pytest_cache
