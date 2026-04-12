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

#include "py_heterodyne.hpp"

#if ENABLE_ROCM
#include "py_heterodyne_rocm.hpp"
#endif

PYBIND11_MODULE(dsp_heterodyne, m) {
    m.doc() = "dsp::heterodyne — LFM dechirp and mixing\n\n"
              "Classes:\n"
              "  HeterodyneDechirp - LFM dechirp pipeline (OpenCL)\n"
              "  HeterodyneROCm    - LFM dechirp + correct (ROCm)\n";

    register_heterodyne(m);

#if ENABLE_ROCM
    register_heterodyne_rocm(m);
#endif
}
