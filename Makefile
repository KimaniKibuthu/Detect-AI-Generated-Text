start-mlflow:
	mlflow ui

start-streamlit:
	poetry run streamlit run streamlit_app.py

start-fastapi:
	poetry run uvicorn fastapi_app:app --reload

run-full-pipeline:
	python src/pipelines/full_pipeline.py

.PHONY: start-mlflow start-streamlit start-fastapi run-full-pipeline
