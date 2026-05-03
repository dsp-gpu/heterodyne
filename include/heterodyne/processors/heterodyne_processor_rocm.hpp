#pragma once

// ============================================================================
// HeterodyneProcessorROCm — ROCm/HIP реализация IHeterodyneProcessor
//
// ЧТО:    Реализация IHeterodyneProcessor: запускает HIP-ядра
//         dechirp_multiply (s_dc = conj(s_rx · s_ref)) и dechirp_correct
//         (сдвиг f_beat → DC) через hiprtc-скомпилированный модуль.
//         Управляет кэшем GPU-буферов (rx, ref, dc, corr, freq) с
//         re-allocation только при изменении размеров. Используется
//         HeterodyneDechirp как Strategy для ROCm backend'а.
//
// ЗАЧЕМ:  Production-путь dechirp на AMD GPU (RDNA4+ / CDNA). Изолирует
//         HIP-зависимости от публичного API: фасад HeterodyneDechirp
//         работает через IHeterodyneProcessor*, не зная о hipModule_t /
//         hipStream_t. На non-ROCm сборках (Windows) — stub с throw,
//         чтобы код собирался для линковки и Python-биндингов.
//
// ПОЧЕМУ: - Layer 6 Ref03 (Facade-of-Ops): держит GpuContext (kernel
//           compile/cache) и кэш GPU-буферов; фактическая логика — в
//           ядрах (heterodyne_kernels_rocm.hpp).
//         - OPT-1: kernel'ы компилируются ОДИН раз (compiled_ флаг через
//           EnsureCompiled). hiprtc-компиляция дорогая (~50 мс).
//         - OPT-2: GPU-буферы кэшируются (cached_total_, cached_samples_,
//           cached_antennas_). Re-allocation только при изменении размеров —
//           иначе hipMalloc/hipFree на каждый вызов = 100+ мкс overhead.
//         - OPT-3: DechirpWithGPURef — оба входа уже на GPU (нет PCIe для
//           ref). Используется когда reference-сигнал уже сгенерирован
//           на GPU (LfmConjugateGenerator → конвейер без CPU roundtrip).
//         - Move разрешён, copy запрещён (=delete) — owns hipModule +
//           buffers; копирование = chaos с GPU lifetime.
//         - kBlockSize = 256: оптимум для warp=64 на RDNA4 (4 wavefront
//           per block, full SIMD utilization). Меньше → idle threads,
//           больше → register spill.
//         - prof_events перегрузки методов — production-путь без overhead'а
//           сборки событий (nullptr), benchmark-путь — с явным prof_events*.
//         - GpuContext (Ref03 Layer 1) — единая точка для kernel
//           compile/cache; backend_ хранится отдельно для hipMalloc через
//           IBackend::AllocateDevice().
//
// Использование:
//   IBackend* backend = drv.GetBackend();
//   HeterodyneProcessorROCm proc(backend);
//   auto dc = proc.Dechirp(rx_data, ref_data, params);
//   auto cr = proc.Correct(dc, f_beat_hz, params);
//
// История:
//   - Создан:  2026-02-23 (порт HeterodyneProcessorOpenCL → HIP/ROCm)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include "../i_heterodyne_processor.hpp"
#include <core/interface/i_backend.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/profiling_types.hpp>

#include <hip/hip_runtime.h>
#include <utility>
#include <vector>

namespace drv_gpu_lib {

/// ROCm profiling events: (name, ROCmProfilingData) pairs collected during processing.
using HeterodyneROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class HeterodyneProcessorROCm
 * @brief ROCm/HIP реализация IHeterodyneProcessor: dechirp_multiply + dechirp_correct.
 *
 * @note Move разрешён, copy запрещён — owns hipModule + GPU-буферы.
 * @note Требует #if ENABLE_ROCM. На non-ROCm сборках — stub с runtime_error.
 * @note Не thread-safe. Один экземпляр = один владелец GPU-кэша и hipModule.
 * @note OPT-1/2: kernel'ы скомпилированы один раз, GPU-буферы кэшируются
 *       по размерам (cached_total_, cached_samples_, cached_antennas_).
 * @see IHeterodyneProcessor — интерфейс (Strategy).
 * @see GetHeterodyneKernelSource_rocm() — kernel source для hiprtc.
 * @see HeterodyneDechirp — Layer 6 фасад, держит unique_ptr на этот класс.
 */
class HeterodyneProcessorROCm : public IHeterodyneProcessor {
public:
  explicit HeterodyneProcessorROCm(IBackend* backend);
  ~HeterodyneProcessorROCm();

  // No copy
  HeterodyneProcessorROCm(const HeterodyneProcessorROCm&) = delete;
  HeterodyneProcessorROCm& operator=(const HeterodyneProcessorROCm&) = delete;

  // Move support
  HeterodyneProcessorROCm(HeterodyneProcessorROCm&& other) noexcept;
  HeterodyneProcessorROCm& operator=(HeterodyneProcessorROCm&& other) noexcept;

  /**
   * @brief Перегрузка `Dechirp` — wrapper, делегирует в основную с аргументами (rx_data, ref_data, params, nullptr). Формула: s_dc = conj(s_rx * s_ref) на GPU.
   * @see Dechirp (основная перегрузка)
   *
   * @param rx_data CPU-данные [num_antennas × num_samples] complex<float> (flat).
   * @param ref_data Reference [num_samples] complex<float> = conj(s_tx).
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   *
   * @return Dechirp-данные [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) override {
    return Dechirp(rx_data, ref_data, params, nullptr);
  }

  /**
   * @brief Dechirp с CPU-входом и сборкой ROCm-событий профилирования. H2D → multiply kernel → D2H.
   *
   * @param rx_data CPU-данные [num_antennas × num_samples] complex<float> (flat).
   * @param ref_data Reference [num_samples] complex<float> = conj(s_tx).
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr] }
   *
   * @return Dechirp-данные [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& rx_data,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params,
      HeterodyneROCmProfEvents* prof_events);

  /**
   * @brief Перегрузка `Correct` — wrapper, делегирует в основную с аргументами (dc_data, f_beat_hz, params, nullptr). Frequency correction: умножение на exp(j * phase_step * n).
   * @see Correct (основная перегрузка)
   *
   * @param dc_data Dechirp-данные [num_antennas × num_samples] complex<float>.
   * @param f_beat_hz Beat-частоты по антеннам [num_antennas], Гц.
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   *
   * @return Скорректированные данные [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      const HeterodyneParams& params) override {
    return Correct(dc_data, f_beat_hz, params, nullptr);
  }

  /**
   * @brief Частотная коррекция (сдвиг f_beat → DC) с ROCm-профилированием. exp(-j·2π·f_beat·t).
   *
   * @param dc_data Dechirp-данные [num_antennas × num_samples] complex<float>.
   * @param f_beat_hz Beat-частоты по антеннам [num_antennas], Гц.
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr] }
   *
   * @return Скорректированные данные [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& dc_data,
      const std::vector<float>& f_beat_hz,
      const HeterodyneParams& params,
      HeterodyneROCmProfEvents* prof_events);

  /**
   * @brief Перегрузка `DechirpFromGPU` — wrapper, делегирует в основную с аргументами (rx_gpu_ptr, ref_data, params, nullptr). Dechirp из внешнего GPU-буфера (void* = hipDeviceptr_t).
   * @see DechirpFromGPU (основная перегрузка)
   *
   * @param rx_gpu_ptr Внешний GPU-буфер (hipDeviceptr_t) [num_antennas × num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param ref_data Reference [num_samples] complex<float> на CPU (H2D внутри метода).
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   *
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  std::vector<std::complex<float>> DechirpFromGPU(
      void* rx_gpu_ptr,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params) override {
    return DechirpFromGPU(rx_gpu_ptr, ref_data, params, nullptr);
  }

  /**
   * @brief Dechirp с внешним GPU-входом и ROCm-профилированием. Без H2D для rx, есть H2D для ref.
   *
   * @param rx_gpu_ptr Внешний GPU-буфер (hipDeviceptr_t) [num_antennas × num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param ref_data Reference [num_samples] complex<float> на CPU (H2D внутри метода).
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr] }
   *
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  std::vector<std::complex<float>> DechirpFromGPU(
      void* rx_gpu_ptr,
      const std::vector<std::complex<float>>& ref_data,
      const HeterodyneParams& params,
      HeterodyneROCmProfEvents* prof_events);

  /**
   * @brief Перегрузка `DechirpWithGPURef` — wrapper, делегирует в основную с аргументами (rx_gpu_ptr, ref_gpu_ptr, params, nullptr). OPT-3: и rx, и ref уже на GPU.
   * @see DechirpWithGPURef (основная перегрузка)
   *
   * @param rx_gpu_ptr Внешний GPU-буфер с rx [num_antennas × num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param ref_gpu_ptr Внешний GPU-буфер с conj(LFM) ref [num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   *
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  std::vector<std::complex<float>> DechirpWithGPURef(
      void* rx_gpu_ptr,
      void* ref_gpu_ptr,
      const HeterodyneParams& params) override {
    return DechirpWithGPURef(rx_gpu_ptr, ref_gpu_ptr, params, nullptr);
  }

  /**
   * @brief OPT-3 dechirp: rx и ref уже на GPU, без PCIe для ref. С ROCm-профилированием.
   *
   * @param rx_gpu_ptr Внешний GPU-буфер с rx [num_antennas × num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param ref_gpu_ptr Внешний GPU-буфер с conj(LFM) ref [num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr] }
   *
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
  std::vector<std::complex<float>> DechirpWithGPURef(
      void* rx_gpu_ptr,
      void* ref_gpu_ptr,
      const HeterodyneParams& params,
      HeterodyneROCmProfEvents* prof_events);

private:
  void EnsureCompiled();
  void ReleaseGpuResources();

  /** OPT-2: Allocate/reuse GPU buffers when size changes */
  void EnsureBuffers(int total_samples, int num_samples);

  GpuContext ctx_;  ///< Ref03: compilation, stream, disk cache
  IBackend*  backend_ = nullptr;  ///< Non-owning, for EnsureBuffers (hipMalloc)
  bool       compiled_ = false;

  // OPT-2: Cached GPU buffers (reused across calls, size-dependent)
  void*  buf_rx_   = nullptr;
  void*  buf_ref_  = nullptr;
  void*  buf_dc_   = nullptr;
  void*  buf_corr_ = nullptr;
  void*  buf_freq_ = nullptr;
  int    cached_total_    = 0;
  int    cached_samples_  = 0;
  int    cached_antennas_ = 0;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace drv_gpu_lib

#else  // !ENABLE_ROCM

// ═══════════════════════════════════════════════════════════════════════════
// Stub for non-ROCm builds (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#include "../i_heterodyne_processor.hpp"
#include <core/interface/i_backend.hpp>
#include <stdexcept>

namespace drv_gpu_lib {

class HeterodyneProcessorROCm : public IHeterodyneProcessor {
public:
  explicit HeterodyneProcessorROCm(IBackend* /*backend*/) {}
  ~HeterodyneProcessorROCm() = default;

  /**
   * @brief Stub: бросает runtime_error — Dechirp доступен только в ROCm-сборке.
   *
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
  std::vector<std::complex<float>> Dechirp(
      const std::vector<std::complex<float>>& /*rx_data*/,
      const std::vector<std::complex<float>>& /*ref_data*/,
      const HeterodyneParams& /*params*/) override {
    throw std::runtime_error("HeterodyneProcessorROCm::Dechirp: ROCm not enabled");
  }

  /**
   * @brief Stub: бросает runtime_error — Correct доступен только в ROCm-сборке.
   *
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
  std::vector<std::complex<float>> Correct(
      const std::vector<std::complex<float>>& /*dc_data*/,
      const std::vector<float>& /*f_beat_hz*/,
      const HeterodyneParams& /*params*/) override {
    throw std::runtime_error("HeterodyneProcessorROCm::Correct: ROCm not enabled");
  }

  /**
   * @brief Stub: бросает runtime_error — DechirpFromGPU доступен только в ROCm-сборке.
   *
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
  std::vector<std::complex<float>> DechirpFromGPU(
      void* /*rx_gpu_ptr*/,
      const std::vector<std::complex<float>>& /*ref_data*/,
      const HeterodyneParams& /*params*/) override {
    throw std::runtime_error("HeterodyneProcessorROCm::DechirpFromGPU: ROCm not enabled");
  }

  /**
   * @brief Stub: бросает runtime_error — DechirpWithGPURef доступен только в ROCm-сборке.
   *
   *
   * @return Никогда не возвращает (всегда throw).
   *   @test_check throws std::runtime_error
   *
   * @throws std::runtime_error всегда: "ROCm not enabled".
   *   @test_check throws std::runtime_error
   */
  std::vector<std::complex<float>> DechirpWithGPURef(
      void* /*rx_gpu_ptr*/,
      void* /*ref_gpu_ptr*/,
      const HeterodyneParams& /*params*/) override {
    throw std::runtime_error("HeterodyneProcessorROCm::DechirpWithGPURef: ROCm not enabled");
  }
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
