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
 *   HeterodyneROCm    — LFM dechirp + correct (ROCm)
 */

#include "py_helpers.hpp"

#include "py_gpu_context.hpp"
#include "py_heterodyne_rocm.hpp"

PYBIND11_MODULE(dsp_heterodyne, m) {
    m.doc() = "dsp::heterodyne — LFM dechirp and mixing (ROCm)\n\n"
              "Classes:\n"
              "  ROCmGPUContext    - GPU context (AMD ROCm)\n"
              "  HeterodyneROCm   - LFM dechirp + correct (ROCm)\n";

    // ROCmGPUContext зарегистрирован в dsp_core (один раз глобально).
    // Импортируем для гарантии что core загружен перед использованием типа в сигнатурах.
    py::module_::import("dsp_core");

    register_heterodyne_rocm(m);
}
