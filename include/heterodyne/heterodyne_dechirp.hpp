#pragma once

// ============================================================================
// HeterodyneDechirp — публичный фасад LFM dechirp pipeline (Layer 6 Ref03)
//
// ЧТO:    Главный публичный API модуля heterodyne. Координирует полный
//         pipeline LFM-дешифровки: генерация conj(LFM) → dechirp multiply
//         → FFT → поиск пика (f_beat) → частотная коррекция → проверка.
//         По f_beat вычисляет дальность R = c·T·f_beat / (2·B). Держит
//         IHeterodyneProcessor (Strategy: ROCm/OpenCL) и кэшированный
//         LfmConjugateGeneratorROCm (OPT-4).
//
// ЗАЧЕМ:  Это публичный API модуля — Python-биндинги (dsp_heterodyne)
//         работают через HeterodyneDechirp::Process / ProcessExternal.
//         Один класс координирует все стадии (Controller GRASP), не
//         делая kernel-launch'и сам — делегирует Strategy. ProcessExternal
//         — для интеграции с внешними OpenCL/ROCm-программами (входной
//         буфер уже на GPU, без PCIe-копирования).
//
// ПОЧЕМУ: - Layer 6 Ref03 (Facade): тонкий координатор, не содержит kernel-
//           логики. Все вычисления — в IHeterodyneProcessor реализациях.
//         - Move/copy запрещены (=delete) — owns Strategy + кэшированный
//           генератор + last_result; копирование = chaos с GPU lifetime.
//         - OPT-4: LfmConjugateGenerator кэшируется (params_dirty_ флаг).
//           Пересоздаётся ТОЛЬКО при SetParams — иначе при каждом Process
//           генерация conj(LFM) занимала бы ~1 мс.
//         - BackendType compute_backend (default ROCm): Strategy выбирается
//           в конструкторе. OpenCL-путь оставлен для legacy-интеграций
//           (ROCm-only — общий принцип, но heterodyne исторически имел
//           OpenCL implementation для совместимости).
//         - ProcessExternal принимает void* — это либо cl_mem, либо
//           hipDeviceptr_t, Strategy-реализация интерпретирует.
//           Внешняя программа владеет указателем — фасад НЕ освобождает.
//         - last_result_ кэшируется (GetLastResult) — для повторного
//           чтения без re-Process (Python tests / debugging).
//
// Использование:
//   auto& backend = drv.GetBackend();
//   HeterodyneDechirp het(&backend);
//   HeterodyneParams params{.f_start=0, .f_end=1e6f, .sample_rate=12e6f,
//                            .num_samples=4000, .num_antennas=5};
//   het.SetParams(params);
//   auto result = het.Process(rx_matrix);  // [num_antennas * num_samples]
//   for (auto& a : result.antennas) {
//     std::cout << "ant=" << a.antenna_idx << " R=" << a.range_m << " m\n";
//   }
//
// История:
//   - Создан:  2026-02-21 (LFM dechirp facade: ROCm + OpenCL Strategy)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include "i_heterodyne_processor.hpp"
#include "heterodyne_params.hpp"
#include <core/common/backend_type.hpp>
#include <signal_generators/generators/lfm_conjugate_generator_rocm.hpp>
#include <signal_generators/params/signal_request.hpp>
#include <signal_generators/params/system_sampling.hpp>
#include <memory>

namespace drv_gpu_lib {

class IBackend;

/**
 * @class HeterodyneDechirp
 * @brief Layer 6 Ref03 фасад LFM-дешифровки: dechirp → FFT → peak → range.
 *
 * @note Move/copy запрещены — owns Strategy + cached LFM generator + last_result.
 * @note Lifecycle: ctor(backend, backend_type) → SetParams(params) → Process*.
 * @note OPT-4: conj(LFM) генератор кэшируется, пересоздаётся только в SetParams.
 * @note Не thread-safe. Один экземпляр = один владелец Strategy + GPU-кэша.
 * @see IHeterodyneProcessor — Strategy (ROCm-реализация).
 * @see HeterodyneParams — параметры LFM (f_start, f_end, sample_rate, ...).
 * @see HeterodyneResult — per-antenna результаты (f_beat, range, SNR).
 * @ingroup grp_heterodyne
 */
class HeterodyneDechirp {
public:
  /**
   * @param backend         Pointer to DrvGPU backend (does not own)
   * @param compute_backend OpenCL (default) or ROCm
   */
  explicit HeterodyneDechirp(
      IBackend* backend,
      BackendType compute_backend = BackendType::ROCm);

  HeterodyneDechirp(const HeterodyneDechirp&) = delete;
  HeterodyneDechirp& operator=(const HeterodyneDechirp&) = delete;

  /** Set LFM parameters (must be called before Process) */
  void SetParams(const HeterodyneParams& params);

  /**
   * Full pipeline from CPU data.
   * Input: s_rx flat matrix [num_antennas * num_samples], complex float
   *
   * Pipeline:
   *   1. Generate conj(LFM) reference (OPT-4: cached)
   *   2. Dechirp multiply: s_dc = s_rx * conj(s_tx) (OPT-3: GPU ref)
   *   3. FFT -> find peak -> f_beat
   *   4. Calculate range + SNR
   */
  HeterodyneResult Process(
      const std::vector<std::complex<float>>& rx_data);

  /**
   * External GPU buffer variant.
   * rx_gpu_ptr is a pointer to cl_mem (OpenCL) or hipDeviceptr_t (ROCm).
   * External program owns the buffer — NOT freed by HeterodyneDechirp.
   * params are passed from CPU (metadata: fs, B, N, antennas).
   */
  HeterodyneResult ProcessExternal(
      void* rx_gpu_ptr,
      const HeterodyneParams& params);

  /** Last result (cached) */
  const HeterodyneResult& GetLastResult() const { return last_result_; }

  /** Get current params */
  const HeterodyneParams& GetParams() const { return params_; }

private:
  HeterodyneResult BuildResult(
      const std::vector<std::complex<float>>& dc_data,
      const HeterodyneParams& params);

  /** OPT-4: Lazy-init and cache the conjugate LFM generator */
  void EnsureConjugateGenerator();

  std::unique_ptr<IHeterodyneProcessor> processor_;
  IBackend*                             backend_ = nullptr;
  BackendType                           compute_backend_ = BackendType::ROCm;
  HeterodyneParams                      params_;
  HeterodyneResult                      last_result_;

  // OPT-4: Cached conjugate LFM generator (rebuilt only on SetParams)
  std::unique_ptr<signal_gen::LfmConjugateGeneratorROCm> conj_gen_;
  bool params_dirty_ = true;  // true = need to rebuild conj_gen_
};

}  // namespace drv_gpu_lib