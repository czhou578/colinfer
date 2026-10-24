// hc_bind.cpp -- torch bindings of hc.cu. bindings.cpp calls bind_hc(m) from the module. Kept apart from bindings.cpp so that
// the kernel files of the qwen4_exp work can be developed in parallel.
#include <torch/extension.h>

void bind_hc(pybind11::module_& m) {
    (void)m;
}
