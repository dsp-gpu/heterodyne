---
schema_version: 1
kind: use_case
id: heterodyne_basic
repo: heterodyne
title: "Как выполнить гетеродин на GPU"
synonyms:
  ru:
    - "гетеродин для обработки сигналов"
    - "базовая гетеродин обработка"
    - "GPU гетеродин"
    - "сигналы гетеродин на GPU"
    - "простой гетеродин"
    - "обработка сигналов гетеродин"
    - "базовая гетеродинная схема"
    - "гетеродин на ROCm"
  en:
    - "heterodyne processing"
    - "basic heterodyne"
    - "gpu heterodyne"
    - "signal processing heterodyne"
    - "simple heterodyne"
    - "heterodyne signal processing"
    - "basic heterodyne scheme"
    - "heterodyne on ROCm"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - core__hybrid_external_context__usecase__v1
  - core__zero_copy__usecase__v1
  - signal_generators__form_signal_rocm__usecase__v1
maturity: stable
language: cpp
tags: [heterodyne, gpu, dsp, rocm, signal_processing, basic, fft, batch, heterodyne_basic]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Как выполнить гетеродин на GPU

## Когда применять

Для базовой обработки сигналов гетеродина на GPU без сложных модулей. Используется когда требуется простая реализация преобразования сигналов с помощью гетеродинной схемы.

## Решение

Класс — `(unknown)`, метод `(unknown)`.

_Пример кода не найден в `tests/` или `examples/`._

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [core__hybrid_external_context__usecase__v1](./hybrid_external_context.md)
- См. [core__zero_copy__usecase__v1](./zero_copy.md)
- См. [signal_generators__form_signal_rocm__usecase__v1](./form_signal_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/heterodyne/tests/test_heterodyne_basic.hpp:1`
