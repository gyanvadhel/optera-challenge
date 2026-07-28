.PHONY: run install test ablate replay clean

## The one command. Works with no API key (offline simulator).
run:
	python run.py --replay --ablate

install:
	python -m pip install -r requirements.txt

test:
	python -m unittest discover -s tests -v

## Quick smoke test on the first 8 images.
smoke:
	python run.py --offline --limit 8

## Lever-by-lever cost attribution.
ablate:
	python run.py --offline --ablate

## Contact sheets for hand-labelling ground truth.
sheets:
	python tools/contact_sheet.py

clean:
	rm -rf out .optera_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
