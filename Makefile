train:
	PYTHONPATH=./ python src/train --no-manual -r -l

heavy-train:
	PYTHONPATH=./ python src/train --no-manual -r -h
