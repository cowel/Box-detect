train:
	PYTHONPATH=./src python src/train --no-manual -r -l

heavy-train:
	PYTHONPATH=./src python src/train --no-manual -r -h
