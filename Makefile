# Makefile
# description: Install virtual environment and libraries to run course notebooks.

PYTHON_VENV = .venv

init: venv lib pipeline

# Install required libraries. 
lib: 
	@. $(PYTHON_VENV)/bin/activate && pip install \
		fbpca \
		matplotlib \
		nltk \
		numpy \
		sklearn \
		spacy 

# Download spacy pipeline.
pipeline:
	. $(PYTHON_VENV)/bin/activate && python -m spacy download en_core_web_sm

# Create virtual environment.
venv:
	test -d $(PYTHON_VENV) || python3 -m venv $(PYTHON_VENV)

.PHONY: lib pipeline
