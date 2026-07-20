#!/usr/bin/env bash
#
# go_build_check.sh — compile-gate for a generated Go snippet (SPEC.md §12,
# the optional "go build 把片段塞进最小 module 编译通过" leg of the inner loop).
#
# Wraps a few lines of candidate Go code into a throwaway, fully sandboxed
# module in a temp dir and runs `go build` (plus `go vet` if available),
# printing PASS / FAIL. The sandbox is hermetic and OFFLINE:
#   - its own GOCACHE / GOPATH / GOMODCACHE under the temp dir
#   - GOPROXY=off, GOSUMDB=off, GOFLAGS=-mod=mod, GOWORK=off, GOTOOLCHAIN=local
# so it neither touches the network nor the surrounding repo's modules.
#
# Usage:  go_build_check.sh PATH_TO_GO_SNIPPET
#   The snippet may be a bare function (no package line) — we prepend a package
#   clause — or a complete file with its own `package`/`func main`.
#
# Exit codes:  0 = PASS (compiles)   1 = FAIL (build error)   2 = SKIP (no go)
#
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 PATH_TO_GO_SNIPPET" >&2
    exit 1
fi
SNIPPET="$1"
[ -f "${SNIPPET}" ] || { echo "FAIL: snippet file not found: ${SNIPPET}" >&2; exit 1; }

if ! command -v go >/dev/null 2>&1; then
    echo "SKIP: go toolchain not found on PATH — cannot compile-check." >&2
    exit 2
fi

# Hermetic temp sandbox (self-cleaning).
TMPDIR_SB="$(mktemp -d "${TMPDIR:-/tmp}/go_build_check.XXXXXX")"
cleanup() { rm -rf "${TMPDIR_SB}"; }
trap cleanup EXIT

export GOCACHE="${TMPDIR_SB}/gocache"
export GOPATH="${TMPDIR_SB}/gopath"
export GOMODCACHE="${TMPDIR_SB}/gomodcache"
export GOPROXY=off
export GOSUMDB=off
export GOFLAGS=-mod=mod
export GOWORK=off            # ignore any parent go.work
export GOTOOLCHAIN=local     # never auto-download a toolchain (offline)
export GO111MODULE=on

# Place the snippet. If it already declares a package, keep it verbatim;
# otherwise wrap it as a library package so a bare function compiles without
# needing a func main().
DST="${TMPDIR_SB}/snippet.go"
if grep -qE '^[[:space:]]*package[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "${SNIPPET}"; then
    cp "${SNIPPET}" "${DST}"
else
    { printf 'package sandboxcheck\n\n'; cat "${SNIPPET}"; } > "${DST}"
fi

# Build inside the sandbox.
(
    cd "${TMPDIR_SB}"
    go mod init go-onebit-buildcheck >/dev/null 2>&1 || true
)

BUILD_LOG="${TMPDIR_SB}/build.log"
if ( cd "${TMPDIR_SB}" && go build ./... ) >"${BUILD_LOG}" 2>&1; then
    # Build OK. go vet is advisory — report but do not fail the gate on it.
    VET_LOG="${TMPDIR_SB}/vet.log"
    if ( cd "${TMPDIR_SB}" && go vet ./... ) >"${VET_LOG}" 2>&1; then
        echo "PASS: ${SNIPPET} compiles (go build + go vet clean)"
    else
        echo "PASS: ${SNIPPET} compiles (go build ok; go vet warnings below)"
        sed 's/^/  vet| /' "${VET_LOG}" >&2 || true
    fi
    exit 0
else
    echo "FAIL: ${SNIPPET} does not compile:"
    sed 's/^/  | /' "${BUILD_LOG}" >&2 || true
    exit 1
fi
