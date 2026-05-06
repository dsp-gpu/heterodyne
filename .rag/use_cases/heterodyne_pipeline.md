---
schema_version: 1
kind: use_case
id: heterodyne_pipeline
repo: heterodyne
title: "Обработка сигналов гетеродинного приемника на GPU"
synonyms:
  ru:
    - "Обработка сигналов гетеродинного приемника"
    - "Гетеродинный сигнал на GPU"
    - "Пайплайн обработки сигналов"
    - "Обработка сигналов с преобразованием Фурье"
    - "Многоэтапная обработка сигналов"
    - "Смешивание и фильтрация сигналов"
    - "Обработка сигналов с ROCm"
    - "Гетеродинный приемник на GPU"
  en:
    - "Heterodyne signal processing"
    - "GPU-based heterodyne pipeline"
    - "Signal processing pipeline for antenna arrays"
    - "FFT-based signal processing"
    - "Multi-stage signal processing"
    - "Mixing and filtering signals"
    - "Heterodyne receiver processing"
    - "Batch signal processing with ROCm"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - heterodyne__heterodyne_basic__usecase__v1
  - heterodyne__heterodyne_benchmark_rocm__usecase__v1
  - core__profiling_facade__usecase__v1
maturity: stable
language: cpp
tags: [heterodyne, rocm, signal_processing, gpu, pipeline, fft, antenna_array, batch_processing, dsp, heterodyne_pipeline]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Обработка сигналов гетеродинного приемника на GPU

## Когда применять

Когда требуется обработка сигналов гетеродинного приемника с несколькими этапами обработки на GPU

## Решение

Класс — `(unknown)`, метод `(unknown)`.

_Пример кода не найден в `tests/` или `examples/`._

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [heterodyne__heterodyne_basic__usecase__v1](./heterodyne_basic.md)
- См. [heterodyne__heterodyne_benchmark_rocm__usecase__v1](./heterodyne_benchmark_rocm.md)
- См. [core__profiling_facade__usecase__v1](./profiling_facade.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/heterodyne/tests/test_heterodyne_pipeline.hpp:1`
