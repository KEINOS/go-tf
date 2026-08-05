# Contributing

Thank you for contributing to go-tf.

## Development commands

Run these commands from the repository root.

```bash
make test      # go test -race -cover ./...
make build     # go build ./...
make lint      # golangci-lint + markdownlint + yamlfmt
make lint-fix  # apply automatic fixes where supported
make check     # lint-fix + test + build + module verification
make clean     # clean Go test cache
```

## Test expectations

The current test suite validates stable parameter registration, expected tensor shapes, forward pass integration with go-nn tensor embedding, and training checkpoint round trip reproducibility for model and optimizer state.

## Pull request checklist

- Keep model-independent primitives in go-nn.
- Keep model-specific composition and integration tests in go-tf.
- Run `make check` before opening or updating a pull request.
- Keep changes focused and include tests for behavior changes.
