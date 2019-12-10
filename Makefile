train:
	PYTHONPATH=./src python bin/train_model.py --no-manual -r -l -a

heavy-train:
	PYTHONPATH=./src python bin/train_model.py --no-manual -r -h -m

test-train:
	PYTHONPATH=./src python bin/train_model.py --no-manual
