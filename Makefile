export TWINE_USERNAME?=__token__
export TWINE_PASSWORD?=your_token

install-poetry::
	python3 -m pip install pip --upgrade
	python3 -m pip install poetry==1.5.0

install-env::
	poetry install --all-extras --ansi --no-root

build::
	poetry run python -m build --sdist --wheel .

linter::
	poetry run pylint allms --reports=no --output-format=colorized --fail-under=8.0

tests::
	poetry run python -m pytest -s --verbose

publish::
	poetry run python -m twine upload --verbose dist/*

docs::
	poetry run mkdocs build
