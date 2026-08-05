GO ?= go
PKGS ?= ./tf/...
GO_TEST_FLAGS ?= -race -cover
GO_BENCH_FLAGS ?= -run '^$$' -bench Benchmark -benchmem -count=1
FUZZ_TIME ?= 3s

define run-check
	@output_file=$$(mktemp); \
	trap 'rm -f "$$output_file"' 0 1 2 3 15; \
	printf '* Running %s ... ' '$(1)'; \
	if { $(2); } >"$$output_file" 2>&1; then \
		printf 'OK\n'; \
	else \
		exit_code=$$?; \
		printf 'FAILED\n'; \
		cat "$$output_file"; \
		exit "$$exit_code"; \
	fi
endef

.PHONY: all
all: test

.PHONY: build
build:
	$(call run-check,go build,$(GO) build $(PKGS))

.PHONY: test
test:
	$(call run-check,go test,$(GO) test $(GO_TEST_FLAGS) $(PKGS))

.PHONY: bench
bench:
	$(call run-check,go benchmark,$(GO) test $(PKGS) $(GO_BENCH_FLAGS))

.PHONY: fuzz
fuzz: fuzz-new-model fuzz-forward

.PHONY: fuzz-new-model
fuzz-new-model:
	$(call run-check,fuzz new model dimensions,$(GO) test $(PKGS) -run '^$$' -fuzz '^FuzzNewModelDimensions$$' -fuzztime $(FUZZ_TIME))

.PHONY: fuzz-forward
fuzz-forward:
	$(call run-check,fuzz forward indices,$(GO) test $(PKGS) -run '^$$' -fuzz '^FuzzModelForwardIndices$$' -fuzztime $(FUZZ_TIME))

.PHONY: lint
lint: lint-go lint-md lint-yaml

.PHONY: lint-go
lint-go:
	$(call run-check,golangci-lint,golangci-lint run)

.PHONY: lint-md
lint-md:
	$(call run-check,markdownlint,markdownlint-cli2 --config .markdownlint-cli2.yaml '**/*.md')

.PHONY: lint-yaml
lint-yaml:
	$(call run-check,yamlfmt -lint,yamlfmt -conf .yamlfmt -lint .)

.PHONY: lint-fix
lint-fix: lint-go-fix lint-md-fix lint-yaml-fix

.PHONY: lint-go-fix
lint-go-fix:
	$(call run-check,go fix,$(GO) fix $(PKGS))
	$(call run-check,golangci-lint --fix,golangci-lint run --fix)

.PHONY: lint-md-fix
lint-md-fix:
	$(call run-check,markdownlint --fix,markdownlint-cli2 --fix --config .markdownlint-cli2.yaml '**/*.md')

.PHONY: lint-yaml-fix
lint-yaml-fix:
	$(call run-check,yamlfmt,yamlfmt -conf .yamlfmt .)

.PHONY: mod-verify
mod-verify:
	$(call run-check,go mod verify,$(GO) mod verify)

.PHONY: mod-tidy
mod-tidy:
	$(call run-check,go mod tidy -diff,$(GO) mod tidy -diff)

.PHONY: check
check: lint-fix test build mod-verify mod-tidy

.PHONY: clean
clean:
	$(GO) clean -testcache
