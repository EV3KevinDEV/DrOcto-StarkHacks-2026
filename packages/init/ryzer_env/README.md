# Local ryzer_env

This package is a minimal local pass-through `ryzer_env` layer.

Ryzers prepends `ryzer_env` automatically during builds, so this package exists only to satisfy local package resolution when building from this repository root.

## Build and run

```bash
ryzers build doc_ock --base_path . --init_image <upstream-lerobot-image>
ryzers run --name doc-ock-ryzers
```
