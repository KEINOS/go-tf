# Security Policy

## Supported versions

Only the latest commit on `main` and the latest published `v0.0.x` release receive security fixes. Pre-release APIs and checkpoint formats may change while a fix is developed.

## Reporting a vulnerability

Use the repository's private GitHub security advisory reporting flow. Do not disclose a vulnerability in a public issue, discussion, pull request, or test fixture.

Include affected versions, required environment or hardware, reproduction steps, impact, and any known mitigation. Do not include credentials, private keys, personal data, or unrelated production data.

## Untrusted checkpoints

Checkpoint readers verify the format version, declared lengths, element and name limits, complete input length, and SHA-256 checksum before restoring any state. Keep decode limits appropriate for the deployment instead of increasing them solely to accept an unexpected file. A failed decode or restore must be treated as untrusted input and leaves the existing model and optimizer state unchanged.
