# ValueEval'24 Hierocles of Alexandria Development Notes

```shell
poetry install --with development
poetry run python -m valueeval24_hierocles_of_alexandria --help
poetry run python -m valueeval24_hierocles_of_alexandria data/examples/simple.tsv
```

## Running unittests (automatically on push)

```shell
poetry run python -m unittest
```

## Running linter (automatically on push)

```shell
poetry run flake8 src --count --max-complexity=10 --max-line-length=127 --statistics
```

## Release new version

- Change `version` in [`pyproject.toml`](../pyproject.toml)
- Add a release via [Github web interface](https://github.com/touche-webis-de/valueeval24-hierocles-of-alexandria/releases/new), tagged `v<VERSION>`
