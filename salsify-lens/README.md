To run:
Install with:
poetry update
poetry shell
poetry install
create folder/file at (if not exists): saves/logs/main.log

Run:
cd into auto_safe folder
- Run annotated model with:
poetry run python object_detection.py
(on model_to_use var change default model input)
- Train new annotated model with:
poetry run python train_annotated.py.py
- Train unsupervisioned model with:
poetry run train_model.py

Notes:
on 1st run will ask for camera permission and fail. Allow and re-run.
command+c to cancel run
