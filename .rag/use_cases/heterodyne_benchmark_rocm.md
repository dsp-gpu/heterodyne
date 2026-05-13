---
schema_version: 1
kind: use_case
id: heterodyne_benchmark_rocm
repo: heterodyne
title: "Как выполнить бенчмарк гетеродина на GPU"
synonyms:
  ru:
    - "тест производительности гетеродина"
    - "оценка скорости обработки сигналов"
    - "benchmark гетеродинного преобразования"
    - "тестирование ROCm для радиолокации"
    - "анализ антенн на GPU"
    - "обработка сигналов в батчах"
    - "benchmark для массива антенн"
    - "производительность гетеродина на ROCm"
  en:
    - "heterodyne benchmark"
    - "fft for antenna array"
    - "signal processing benchmark"
    - "rocmlib gpu testing"
    - "batch processing benchmark"
    - "antenna array performance"
    - "real-time signal analysis"
    - "gpu acceleration benchmark"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - spectrum__lch_farrow_rocm__usecase__v1
  - spectrum__lch_farrow_benchmark_rocm__usecase__v1
  - heterodyne__heterodyne_basic__usecase__v1
maturity: stable
language: cpp
tags: [heterodyne, rocm, fft, batch, antenna_array, signal_processing, benchmark, gpu_computing, dsp, rocmlib]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Как выполнить бенчмарк гетеродина на GPU

## Когда применять

Когда нужно протестировать производительность гетеродинного преобразования на ROCm с использованием массива антенн и обработкой сигналов в реальном времени

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  std::cout << "\n"
            << "============================================================\n"
            << "  Heterodyne Benchmark (Dechirp / Correct) — ROCm\n"
            << "============================================================\n";

  // Проверить AMD GPU
  if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
    std::cout << "  [SKIP] No AMD GPU available\n";
    return 0;
  }

  try {
    // ── ROCm backend init ─────────────────────────────────────────────────
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    // ── Параметры гетеродина ──────────────────────────────────────────────
    dsp::heterodyne::HeterodyneParams params;
    params.num_antennas = 5;
    params.num_samples  = 4000;
    params.sample_rate  = 12e6f;
    params.f_start      = 0.0f;
    params.f_end        = 1e6f;

    const int total = params.num_antennas * params.num_samples;

    // ── Тестовые данные (pure real = 1+0i, для бенчмарка данные не критичны)
    const std::vector<std::complex<float>> rx_data(total, {1.0f, 0.0f});
    const std::vector<std::complex<float>> ref_data(params.num_samples, {1.0f, 0.0f});
    const std::vector<std::complex<float>> dc_data(total, {1.0f, 0.0f});
// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [spectrum__lch_farrow_rocm__usecase__v1](./lch_farrow_rocm.md)
- См. [spectrum__lch_farrow_benchmark_rocm__usecase__v1](./lch_farrow_benchmark_rocm.md)
- См. [heterodyne__heterodyne_basic__usecase__v1](./heterodyne_basic.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/heterodyne/tests/test_heterodyne_benchmark_rocm.hpp:1`
