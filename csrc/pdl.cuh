// pdl.cuh -- programmatic dependent launch (PLAN.md Phase 6: overlap across kernel boundaries in the decode step).
//
// Every decode-path kernel calls PDL_TRIGGER() first: a kernel launched after it with the programmatic-serialization
// attribute may then start while it still runs. Such a kernel (the skinny GEMM) issues work that does not depend on
// its predecessor (streaming its weights) and calls PDL_WAIT() before touching anything the predecessor writes or
// reads: griddepcontrol.wait returns once the predecessor grid has completed and its memory is visible. Kernels
// launched without the attribute are unaffected (both instructions are no-ops for them).
#pragma once
#define PDL_TRIGGER() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
