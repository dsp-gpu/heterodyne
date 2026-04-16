/**
 * @file dsp_heterodyne_module.cpp
 * @brief pybind11 bindings for dsp::heterodyne
 *
 * Python API:
 *   import dsp_heterodyne
 *   proc = dsp_heterodyne.HeterodyneROCm(ctx)
 *   result = proc.process(signal_data)
 *
 * Экспортируемые классы:
 *   HeterodyneDechirp — LFM dechirp (OpenCL)
 *   HeterodyneROCm    — LFM dechirp + correct (ROCm)
 */

#include "py_helpers.hpp"

// py_heterodyne.hpp — использует GPUContext (OpenCL), только nvidia-ветка
// #include "py_heterodyne.hpp"

#if ENABLE_ROCM
#include "py_gpu_context.hpp"
#include "py_heterodyne_rocm.hpp"
#endif

PYBIND11_MODULE(dsp_heterodyne, m) {
    m.doc() = "dsp::heterodyne — LFM dechirp and mixing (ROCm)\n\n"
              "Classes:\n"
              "  ROCmGPUContext    - GPU context (AMD ROCm)\n"
              "  HeterodyneROCm   - LFM dechirp + correct (ROCm)\n";

#if ENABLE_ROCM
    py::class_<ROCmGPUContext>(m, "ROCmGPUContext",
        "ROCm GPU context (creates HIP backend for AMD GPU).")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("device_name", &ROCmGPUContext::device_name)
        .def_property_readonly("device_index", &ROCmGPUContext::device_index);

    register_heterodyne_rocm(m);
#endif
}
