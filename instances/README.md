# Benchmark instances

The final study uses 1440 deterministic paired instances generated from the protocol in `docs/PROTOCOL.md`.

The repository stores the benchmark manifests and the exact deterministic generator. The full binary `.npz` files can be regenerated locally with:

```bash
python run.py generate --config config/extended_1440_300s.json --root .
```

For the 480-instance paper-size subset:

```bash
python run.py generate --config config/paper_protocol_480_300s.json --root .
```

Each generated file has a canonical SHA-256 hash recorded in the manifest. The recovered generator was verified against all 1440 hashes from the final campaign.
