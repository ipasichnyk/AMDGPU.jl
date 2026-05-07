# CLAUDE.md — Runtime Fidelity Metrics (AMDGPU.jl)

## Project Context
This spec covers changes to AMDGPU.jl that support GPU memory diagnostics used during tensor-network simulation profiling in TensorNetworkQuantumSimulator.jl.

## Before Starting Work
1. Read [specs/runtime-metrics/design.md](design.md) — what changed and why
2. Key file: `src/memory.jl`

## Code Patterns
- Always call `HIP.device_synchronize()` before reading memory stats (async allocations)
- `info()` returns `(free_bytes, total_bytes)` — use `%` not `convert` to avoid overflow on large GPUs
- `memory_snapshot` returns a named tuple so callers can log structured data

## Don't
- Don't read memory stats without synchronizing first — values will be stale
