# go-tf

go-tf is a model-specific Go consumer for Transformer-oriented model composition. It validates model-independent contracts from [go-nn](https://github.com/KEINOS/go-nn).

## Current Scope

This repository currently builds a minimal token-embedding model with go-nn primitives. It checks parameter ownership, forward behavior, optimizer stepping, and training checkpoint round trips before larger Transformer blocks are assembled here.

It does not yet expose a full Transformer architecture. Model-independent neural-network primitives stay in go-nn; this repository keeps model-specific composition and integration tests.

## Package Overview

Package `tf` currently provides the embedding-stage model surface:

- `NewModel`: creates a minimal token embedding model using go-nn module primitives.
- `Forward`: runs embedding lookup for token indices.
- `NewOptimizer`: creates Tensor Adam bound to model parameters.
- `StateDict` and `LoadStateDict`: snapshot and restore model parameter state.
- `WriteTrainingCheckpoint` and `RestoreTrainingCheckpoint`: persist and restore full training state, including optimizer and execution context.
- `Close`: releases model-owned parameter state.

The scopes are:

- Keep model-independent neural-network primitives in go-nn.
- Keep model-specific composition and integration tests in go-tf.

## Roadmap

The planned path from the current embedding consumer to a minimal Transformer is:

1. Keep the token embedding model stable as the integration baseline.
2. Add model configuration APIs before constructor arguments grow.
3. Consume positional embedding, attention, normalization, MLP, and loss primitives from go-nn as they become available.
4. Assemble a minimal Transformer block and training example in go-tf without duplicating reusable primitives from go-nn.
5. Keep checkpoints reproducible across model, optimizer, execution mode, and generator state.

## Requirements

- Go 1.26.5 or newer

## Usage

```bash
# Install module
go get "github.com/KEINOS/go-tf@latest"
```

```go
// Use module
import "github.com/KEINOS/go-tf/tf"
```

Create a model and run a forward pass:

```go
backend, err := nn.NewTensorBackend(nn.UseCPU)
if err != nil {
    log.Fatal(err)
}
defer backend.Close()

model, err := tf.NewModel(backend, 4, 3, nn.NewGenerator(42, 0))
if err != nil {
    log.Fatal(err)
}
defer model.Close()

out, err := model.Forward([]int{2, 1}, 2)
if err != nil {
    log.Fatal(err)
}

fmt.Println(out.Value().Shape()) // [2 3]
```

Snapshot parameters and load them into another model:

```go
src, err := tf.NewModel(backend, 4, 3, nn.NewGenerator(1, 0))
if err != nil {
    log.Fatal(err)
}
defer src.Close()

dst, err := tf.NewModel(backend, 4, 3, nn.NewGenerator(2, 0))
if err != nil {
    log.Fatal(err)
}
defer dst.Close()

state, err := src.StateDict()
if err != nil {
    log.Fatal(err)
}
if err := dst.LoadStateDict(state); err != nil {
    log.Fatal(err)
}

loaded, err := dst.StateDict()
if err != nil {
    log.Fatal(err)
}
fmt.Println(len(loaded.Entries())) // 1
```

Save and restore training checkpoint state:

```go
optimizer, err := model.NewOptimizer(backend, nn.NewTensorAdamConfig())
if err != nil {
    log.Fatal(err)
}
defer optimizer.Close()

ctx, err := nn.NewExecutionContext(nn.Training, nn.NewGenerator(42, 0))
if err != nil {
    log.Fatal(err)
}

var checkpoint bytes.Buffer
if err := model.WriteTrainingCheckpoint(&checkpoint, optimizer, ctx); err != nil {
    log.Fatal(err)
}

restored, err := model.RestoreTrainingCheckpoint(
    bytes.NewReader(checkpoint.Bytes()),
    backend,
    optimizer,
    nn.DefaultCheckpointLimits(),
)
if err != nil {
    log.Fatal(err)
}

fmt.Println(restored.Mode() == nn.Training) // true
```

## Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for development commands, test expectations, and the contribution workflow.

## Security

See [.github/SECURITY.md](.github/SECURITY.md).

## License

Copyright (c) 2026 KEINOS and go-tf contributors.

This project is licensed under the MIT License. See [LICENSE](LICENSE).
