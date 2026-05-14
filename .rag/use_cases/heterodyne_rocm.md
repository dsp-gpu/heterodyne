---
schema_version: 1
kind: use_case
id: heterodyne_rocm
repo: heterodyne
title: "Heterodyne Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: PyHeterodyneROCm
primary_method: PyHeterodyneROCm
related_classes:
  - strategies::all_maxima_pipeline_rocm
  - heterodyne::heterodyne_processor_rocm
  - linalg::capon_processor
  - stats::statistics_processor
  - spectrum::spectrum_processor_rocm
related_use_cases:
  - heterodyne__heterodyne_benchmark_rocm__usecase__v1
  - heterodyne__heterodyne_pipeline__usecase__v1
  - heterodyne__heterodyne_basic__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Heterodyne Rocm

## Когда применять

_LLM-fallback: см. описание класса._

## Решение

Класс — `PyHeterodyneROCm`, метод `PyHeterodyneROCm`.

```cpp
  int gpu_id = 0;

  ROCmBackend backend;
  try { backend.Initialize(gpu_id); }
  catch (...) { return; }

  TestRunner runner(&backend, "Heterodyne ROCm", gpu_id);

  auto ref = GenerateConjRefCPU();

  // ── Test 1: Single antenna ────────────────────────────────────
  runner.test("single_antenna", [&]() {
    auto rx = GenerateRxFlatCPU({100.f});
    HeterodyneParams p; p.f_start=F_START; p.f_end=F_END;
    p.sample_rate=FS; p.num_samples=N; p.num_antennas=1;
    HeterodyneProcessorROCm proc(&backend);
    auto dc = proc.Dechirp(rx, ref, p);
    float f = CpuFindPeakFrequency(dc, 0, N, FS);
    return ScalarAbsError(static_cast<double>(f), static_cast<double>(MU*100e-6f),
                          F_BEAT_TOL_HZ, "f_beat");
  });

  // ── Test 2: 5 antennas ────────────────────────────────────────
  runner.test("five_antennas", [&]() {
    auto rx = GenerateRxFlatCPU(DELAYS_LINEAR_US);
    HeterodyneParams p; p.f_start=F_START; p.f_end=F_END;
    p.sample_rate=FS; p.num_samples=N; p.num_antennas=ANTENNAS;
    HeterodyneProcessorROCm proc(&backend);
    auto dc = proc.Dechirp(rx, ref, p);
    return ValidateFBeat("five_antennas", dc, DELAYS_LINEAR_US, ANTENNAS);
// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [heterodyne__heterodyne_benchmark_rocm__usecase__v1](./heterodyne_benchmark_rocm.md)
- См. [heterodyne__heterodyne_pipeline__usecase__v1](./heterodyne_pipeline.md)
- См. [heterodyne__heterodyne_basic__usecase__v1](./heterodyne_basic.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/heterodyne/tests/test_heterodyne_rocm.hpp:1`
