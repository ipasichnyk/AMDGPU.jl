# Design: GPU Memory Diagnostics

## Problem

`AMDGPU.info()` used `convert(Int, ...)` on `Csize_t` values returned by `hipMemGetInfo`. On large GPUs (MI300X, 192 GB), these values exceed `typemax(Int32)` and can produce negative results depending on the Julia build. Additionally, there was no structured way to take a synchronized memory snapshot at semantic boundaries (between Trotter layers, after BP update, etc.).

## Changes

### 1. `info()` overflow fix

**File:** `src/memory.jl`

```julia
# before
return convert(Int, free_ref[]), convert(Int, total_ref[])

# after
f = free_ref[] % Int
t = total_ref[] % Int
return max(f, 0), max(t, 0)
```

`% Int` reinterprets the bits rather than range-checking, avoiding the `InexactError` on large values. `max(..., 0)` guards against the rare case where the reinterpretation produces a negative value due to sign bit on 32-bit builds.

---

### 2. `memory_snapshot`

**File:** `src/memory.jl`

Prints a synchronized snapshot of device and pool memory and returns a named tuple for structured logging:

```julia
function memory_snapshot(io=stdout; label=nothing, sync=true)
    sync && HIP.device_synchronize()
    free_bytes, total_bytes = info()
    used_bytes = total_bytes - free_bytes
    pool = Mem.pool_create(AMDGPU.device())
    pool_used    = HIP.used_memory(pool)
    pool_reserved = HIP.reserved_memory(pool)
    non_pool = max(0, used_bytes - pool_reserved)
    # prints device used / total / free, pool used / reserved / non-pool
    return (; device_used, device_free, device_total, pool_used, pool_reserved, non_pool)
end
```

**Purpose:** Allows tensor-network callers to log GPU memory at defined checkpoints (per layer, per gate) without manually synchronizing or computing pool vs device breakdown.

**Used by:** `TensorNetworkQuantumSimulator._gpu_mem_mb()` hook reads a subset of these values for per-gate profiling output.
