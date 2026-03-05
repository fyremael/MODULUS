# Install Flow

## Core Install

```bash
python -m pip install -e .
```

## Development Install

```bash
python -m pip install -e ".[dev]"
```

## Examples Install

```bash
python -m pip install -e ".[examples]"
```

## Report Install

```bash
python -m pip install -e ".[report]"
```

## Full Install

```bash
python -m pip install -e ".[all]"
```

## Smoke Checks

```bash
python -m ruff check modulus scripts
python scripts/generate_api_docs.py --check
python -m pytest
python -m modulus.tests.test_hyperball
python -m modulus.tests.test_groups
python -m modulus.tests.test_lora_grad
```
