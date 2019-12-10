train:
	PYTHONPATH=./src python bin/train_model.py --no-manual -r -l -a

heavy-train:
	PYTHONPATH=./src python src/train --no-manual -r -h

test-train:
	PYTHONPATH=./src python src/train --no-manual
