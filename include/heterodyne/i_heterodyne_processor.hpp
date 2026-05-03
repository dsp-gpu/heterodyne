#pragma once

// ============================================================================
// IHeterodyneProcessor — Strategy-интерфейс LFM dechirp процессора
//
// ЧТО:    Pure-virtual интерфейс для GPU-реализаций dechirp-операций:
//         Dechirp (s_dc = s_rx · conj(s_tx)), Correct (сдвиг f_beat → DC),
//         DechirpFromGPU (внешний GPU-буфер), DechirpWithGPURef (оба
//         входа уже на GPU). Реализуется HeterodyneProcessorROCm
//         (продакшн ROCm-путь). Используется HeterodyneDechirp фасадом
//         через std::unique_ptr<IHeterodyneProcessor>.
//
// ЗАЧЕМ:  Strategy (GoF) — фасад HeterodyneDechirp зависит от абстракции,
//         а не от конкретной ROCm/OpenCL реализации. Это позволяет
//         подменять backend без изменения публичного API (Python-биндинги
//         продолжают работать). Также оставляет место под будущие
//         реализации (CUDA/CPU emulation для unit-тестов).
//
// ПОЧЕМУ: - Все методы, кроме DechirpWithGPURef, pure virtual (=0):
//           бизнес-логика обязательна, реализация её default'ом запретна.
//         - DechirpWithGPURef имеет default-throw реализацию (OPT-3
//           опциональная оптимизация: если backend не поддерживает GPU-ref,
//           фасад fallback'ится на DechirpFromGPU).
//         - Сигнатуры одинаковые на CPU-вход (vector<complex<float>>) для
//           простоты Python-биндингов; GPU-варианты принимают void* —
//           это либо cl_mem (OpenCL), либо hipDeviceptr_t (ROCm),
//           backend сам интерпретирует.
//         - ref_data передаётся отдельно, а не вычисляется внутри:
//           HeterodyneDechirp кэширует conj(LFM)-генератор (OPT-4) — не
//           пересчитывает при каждом Process.
//         - Внешний GPU-буфер — НЕ освобождается процессором (внешняя
//           программа владеет cl_mem / hipDeviceptr_t).
//
// Использование:
//   class MyProcessor : public IHeterodyneProcessor {
//     std::vector<std::complex<float>> Dechirp(...) override { ... }
//     std::vector<std::complex<float>> Correct(...) override { ... }
//     std::vector<std::complex<float>> DechirpFromGPU(...) override { ... }
//     // DechirpWithGPURef — опционально (default бросает runtime_error)
//   };
//
// История:
//   - Создан:  2026-02-21 (Strategy для dechirp: OpenCL + ROCm реализации)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include "heterodyne_params.hpp"
#include <vector>
#include <complex>
#include <stdexcept>

namespace drv_gpu_lib {

/**
 * @class IHeterodyneProcessor
 * @brief Strategy-интерфейс LFM dechirp: Dechirp / Correct / DechirpFromGPU.
 *
 * @note Pure-virtual интерфейс (кроме DechirpWithGPURef — default throws).
 * @note GPU-варианты принимают void* — cl_mem (OpenCL) или hipDeviceptr_t (ROCm).
 * @note Внешний GPU-буфер не освобождается процессором (caller owns).
 * @see HeterodyneProcessorROCm — продакшн-реализация на HIP/ROCm.
 * @see HeterodyneDechirp — Layer 6 фасад, держит unique_ptr<IHeterodyneProcessor>.
 */
class IHeterodyneProcessor {
public:
  virtual ~IHeterodyneProcessor() = default;

  /**
   * @brief Dechirp: s_dc = s_rx · conj(s_tx) на GPU.
   * @param rx_data   Матрица [num_antennas × num_samples] complex float (flat).
   * @param ref_data  Вектор [num_samples] complex float = conj(s_tx).
   * @param params    Параметры LFM.
   *   @test_ref HeterodyneParams
   * @return          Dechirp-матрица [num_antennas × num_samples] complex float.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  virtual std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) = 0;

  /**
   * @brief Частотная коррекция: умножить на exp(-j·2π·f_beat·t) по антеннам.
   * @param dc_data    Dechirp-данные [num_antennas × num_samples].
   * @param f_beat_hz  Beat-частоты по антеннам [num_antennas], Гц.
   * @param params     Параметры LFM.
   *   @test_ref HeterodyneParams
   * @return           Скорректированные данные [num_antennas × num_samples].
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  virtual std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      const HeterodyneParams& params) = 0;

  /**
   * @brief Dechirp с внешним GPU-буфером (caller владеет указателем!).
   *
   * Для интеграции с внешними OpenCL/ROCm-программами: входной буфер
   * уже на GPU (cl_mem или hipDeviceptr_t), процессор НЕ освобождает его.
   *
   * @param rx_cl_mem  Внешний GPU-буфер [num_antennas × num_samples].
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param ref_data   Reference (CPU → GPU внутри метода).
   * @param params     Параметры LFM.
   *   @test_ref HeterodyneParams
   * @return           Dechirp-данные на CPU.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  virtual std::vector<std::complex<float>> DechirpFromGPU(
      void* rx_cl_mem,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) = 0;

  /**
   * @brief OPT-3: Dechirp с rx и ref уже на GPU (без PCIe для ref).
   * @note Default-реализация бросает runtime_error — поддержка опциональна.
   * @throws std::runtime_error если backend не поддерживает GPU-ref.
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex float.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   *   @test_check throws std::runtime_error в default-реализации
   */
  virtual std::vector<std::complex<float>> DechirpWithGPURef(
      void* /*rx_cl_mem*/, void* /*ref_cl_mem*/,
      const HeterodyneParams& /*params*/) {
    throw std::runtime_error("DechirpWithGPURef: not implemented");
  }
};

}  // namespace drv_gpu_lib