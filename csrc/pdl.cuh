// pdl.cuh -- programmatic dependent launch (PLAN.md Phase 6: overlap across kernel boundaries in the decode step).
//
// Each decode-path kernel calls PDL_TRIGGER() first. Then a kernel launched after it with the programmatic-serialization
// attribute can start while it still runs. Such a kernel (the skinny GEMM) issues the work that does not depend on its
// predecessor: it streams its weights. It calls PDL_WAIT() before it touches anything that the predecessor writes or
// reads. griddepcontrol.wait returns when the predecessor grid has completed and its memory is visible. For kernels
// launched without the attribute, both instructions are no-ops.
#pragma once
#define PDL_TRIGGER() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
