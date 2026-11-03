# Builds every example under examples/*.cpp into $(BIN)/cpp-<name>.
ifeq ($(origin CXX),default)
CXX := clang++
endif
CXXFLAGS ?= -std=c++17 -O2 -pthread -Wall -Wextra
BIN      ?= ../../bin
INCLUDES  = -Iinclude -Ithird_party

# Some macOS Command Line Tools installs ship an incomplete toolchain libc++ include dir while
# the SDK has the full one; fall back to the SDK headers when <algorithm> is not found.
LIBCXX_OK := $(shell echo '\#include <algorithm>' | $(CXX) $(CXXFLAGS) -x c++ -fsyntax-only - >/dev/null 2>&1 && echo yes)
ifneq ($(LIBCXX_OK),yes)
SDKROOT := $(shell xcrun --show-sdk-path 2>/dev/null)
ifneq ($(SDKROOT),)
CXXFLAGS += -nostdinc++ -isystem $(SDKROOT)/usr/include/c++/v1
endif
endif

EXAMPLES := $(patsubst examples/%.cpp,%,$(wildcard examples/*.cpp))
TARGETS  := $(addprefix $(BIN)/cpp-,$(EXAMPLES))

.PHONY: all clean
all: $(TARGETS)

$(BIN)/cpp-%: examples/%.cpp include/mep/mep.hpp third_party/nlohmann/json.hpp
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(TARGETS)
