install:
	pip install -r requirements.txt

run-all:
	python run_pipeline.py
	
test:
	pytest test_pipeline.py
