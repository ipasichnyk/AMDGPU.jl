# Implementation: GPU Memory Diagnostics

## Status: Done

## Implemented

- [x] **`info()` overflow fix** — `src/memory.jl`: `convert(Int, ...)` → `% Int` with `max(..., 0)` guard
- [x] **`memory_snapshot`** — `src/memory.jl`: synchronized device + pool memory snapshot with structured return value

## Files Changed

| File | Change |
|------|--------|
| `src/memory.jl` | `info()` overflow fix; added `memory_snapshot` function |
