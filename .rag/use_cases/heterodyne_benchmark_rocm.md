---
schema_version: 1
kind: use_case
id: heterodyne_benchmark_rocm
repo: heterodyne
title: "Heterodyne Benchmark Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - heterodyne__heterodyne_basic__usecase__v1
  - stats__statistics_rocm__usecase__v1
  - spectrum__filters_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Heterodyne Benchmark Rocm

## Когда применять

_LLM-fallback: см. описание класса._

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
    ::dsp::heterodyne::HeterodyneParams params;
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

- См. [heterodyne__heterodyne_basic__usecase__v1](./heterodyne_basic.md)
- См. [stats__statistics_rocm__usecase__v1](./statistics_rocm.md)
- См. [spectrum__filters_benchmark_rocm__usecase__v1](./filters_benchmark_rocm.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/heterodyne/tests/test_heterodyne_benchmark_rocm.hpp:1`
