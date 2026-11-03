# monorl C++ environment SDK (MEP/1)

Header-only, C++17, no dependencies beyond POSIX sockets and the vendored JSON library.

- `include/mep/mep.hpp` — `mep::Env`, typed structs, session manager, `serve_stdio`, `serve_http`, `run_main`.
- `third_party/nlohmann/json.hpp` — [nlohmann/json](https://github.com/nlohmann/json) **v3.12.0**
  single header, downloaded unmodified from
  `https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp` (MIT licence).
- `examples/*.cpp` — `arith` (single-turn) and `number_guess` (multi-turn).

Build: `make` here (or `make -C envs cpp` from the repo root) compiles every example with
`clang++ -std=c++17 -O2 -pthread` into `envs/bin/cpp-<name>`. Override `CXX`/`CXXFLAGS`/`BIN`
on the command line. See `docs/ENV_SDK.md` for the protocol, the API and how to use a server
from a monorl run.
