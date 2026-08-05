# Contributing

Thank you for contributing to go-tf.

## Development commands

Run these commands from the repository root.

```bash
make test      # go test -race -cover ./...
make build     # go build ./...
make lint      # golangci-lint + markdownlint + yamlfmt
make lint-fix  # apply automatic fixes where supported
make bench     # run benchmarks and save output to .bench/bench_*.txt
make fuzz      # run all fuzz targets with bounded fuzz time
make check     # lint-fix + test + build + module verification
make clean     # clean Go test cache
```

## Benchmark workflow for verdict

Use this workflow when you measure before/after performance changes.

1. Run `make bench` before your change and keep the printed path as baseline.
2. Apply your change and run `make bench` again.
3. Compare both outputs with `benchstat` piped to `verdict`:

```bash
benchstat .bench/bench_old.txt .bench/bench_new.txt | verdict
```

`make bench` prints benchmark details (`ns/op`, `B/op`, `allocs/op`) and stores
the same raw output file so it can be reused for `benchstat` and `verdict`.

## Test expectations

The current test suite validates stable parameter registration, expected tensor shapes, forward pass integration with go-nn tensor embedding, and training checkpoint round trip reproducibility for model and optimizer state.

## Pull request checklist

- Keep model-independent primitives in go-nn.
- Keep model-specific composition and integration tests in go-tf.
- Run `make check` before opening or updating a pull request.
- Keep changes focused and include tests for behavior changes.
