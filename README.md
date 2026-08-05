# go-tf

go-tf is a model-specific Transformer consumer in Go. It validates model-independent contracts from [go-nn](https://github.com/KEINOS/go-nn).

## Purpose

This repository builds a small token-embedding model with go-nn primitives. It checks parameter ownership, forward behavior, optimizer stepping, and training checkpoint round trips.

## Scope

- Keep model-independent neural-network primitives in go-nn.
- Keep model-specific composition and integration tests in go-tf.

## Requirements

- Go 1.26.5 or newer
- Network access to download public Go modules from GitHub

## Setup

1. Clone this repository.
2. Download modules and run checks:

```bash
go mod tidy
make check
```

## Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for development commands, test expectations, and the contribution workflow.

## Package Overview

Package `tf` currently provides:

- `NewModel`: creates a minimal token embedding model using go-nn module primitives.
- `Forward`: runs embedding lookup for token indices.
- `NewOptimizer`: creates Tensor Adam bound to model parameters.
- `StateDict` and `LoadStateDict`: snapshot and restore model parameter state.
- `WriteTrainingCheckpoint` and `RestoreTrainingCheckpoint`: persist and restore full training state, including optimizer and execution context.
- `Close`: releases model-owned parameter state.

## Security

See [.github/SECURITY.md](.github/SECURITY.md).
