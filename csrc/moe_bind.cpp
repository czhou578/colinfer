// moe_bind.cpp -- torch bindings of moe.cu. bindings.cpp calls bind_moe(m) from the module. Kept apart from bindings.cpp so that
// the kernel files of the qwen4_exp work can be developed in parallel.
#include <torch/extension.h>

void bind_moe(pybind11::module_& m) {
    (void)m;
}
