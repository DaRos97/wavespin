# Getting Started

## Setup

```bash
git clone https://github.com/DaRos97/wavespin
cd wavespin
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[test]"
```

Run the tests to verify everything works:

```bash
pytest tests/
```

## Running an Example

```bash
python examples/3_staticDispersion.py examples/input_3.txt
```

Example input files are simple `key: value` pairs:

```
Lx: 20
Ly: 20
boundary: periodic
plotLattice: False
```

## Next Steps

- Read the [Physics Background](physics.md) to understand the model.
- See the [User Guide](user-guide/lattice.md) for detailed module documentation.
- Browse the [API Reference](api/lattice.md) for complete function signatures.
