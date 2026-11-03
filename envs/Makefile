# Builds the MEP/1 example environment servers into envs/bin/<lang>-<example>.
#   make go    → go-arith go-wordle go-gridworld   (needs go)
#   make cpp   → cpp-arith cpp-number_guess        (needs clang++)
#   make all   → both;  make clean
BIN := $(abspath bin)
GO  ?= go

GO_EXAMPLES := $(notdir $(wildcard sdk/go/examples/*))

.PHONY: all go cpp clean
all: go cpp

# `go build` is incremental itself, so the target is always invoked.
go:
	@mkdir -p $(BIN)
	cd sdk/go && for ex in $(GO_EXAMPLES); do $(GO) build -o $(BIN)/go-$$ex ./examples/$$ex || exit 1; done

cpp:
	$(MAKE) -C sdk/cpp BIN=$(BIN) all

clean:
	rm -rf $(BIN)
