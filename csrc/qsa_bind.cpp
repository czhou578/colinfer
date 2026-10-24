// qsa_bind.cpp -- torch bindings of qsa.cu. bindings.cpp calls bind_qsa(m) from the module. Kept apart from bindings.cpp so that
// the kernel files of the qwen4_exp work can be developed in parallel.
#include <torch/extension.h>

void bind_qsa(pybind11::module_& m) {
    (void)m;
}
