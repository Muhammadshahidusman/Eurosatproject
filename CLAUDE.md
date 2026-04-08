# Project Guidelines for AI Agents

## Project Overview
This project is a Deep Learning assignment using the EuroSAT dataset, focusing on land cover classification and extending it to Lahore-specific data acquisition.

## Environment & Tooling
- **Package Manager**: `uv` (Use `uv pip install` for dependencies)
- **Python Version**: 3.10
- **Notebooks**: Located in `notebooks/`
- **Data**: Stored in `data/`

## Project Structure
- `notebooks/`: All exploratory and data acquisition notebooks.
- `data/`: Raw and processed datasets.
- `models/`: Model checkpoints and saved weights.
- `src/`: Source code for training, evaluation, and utilities.

## Conventions
- **Notebook Execution**: Use `jupyter nbconvert --to notebook --execute` for headless execution.
- **Dependency Management**: Always use `uv` to ensure consistency with the project's virtual environment.
- **Data Handling**: Use absolute paths or paths relative to the project root.

## Common Tasks
- **Running Data Acquisition**: Run `notebooks/lahore_data_acquisition.ipynb`.
- **Model Training**: Check `src/` for training scripts.
