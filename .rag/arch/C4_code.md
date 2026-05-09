---
schema_version: 1
repo: heterodyne
arch_level: c4
tags:
  - "#level:c4"
  - "#repo:heterodyne"
  - "#layer:compute"
  - "#pattern:Pipeline:HeterodyneProcessorROCm"
  - "#pattern:Pipeline:IHeterodyneProcessor"
  - "#pattern:Bridge:IBackend"
description: "C4 Code — реальные классы с паттернами GoF/SOLID для репо heterodyne."
---

# C4 Code — `heterodyne`

## Классы с паттернами проектирования

| Класс | Паттерн | Brief |
|-------|---------|-------|
| `HeterodyneProcessorROCm` | **Pipeline** |  |
| `IHeterodyneProcessor` | **Pipeline** |  |
| `IBackend` | **Bridge** |  |

## HIP-ядра (`kernels/rocm/`)

*kernels/rocm/ пуст или отсутствует.*

## Все key_classes (FQN список)

- `drv_gpu_lib::HeterodyneProcessorROCm` (24 методов)
- `drv_gpu_lib::HeterodyneDechirp` (8 методов)
- `drv_gpu_lib::HeterodyneResult` (1 методов)
- `drv_gpu_lib::IHeterodyneProcessor` (2 методов)
- `PyHeterodyneDechirp` (6 методов)
- `PyHeterodyneROCm` (5 методов)
- `drv_gpu_lib::HeterodyneParams` (4 методов)
- `test_heterodyne_rocm::HeterodyneCorrectBenchmarkROCm` (3 методов)
- `test_heterodyne_rocm::HeterodyneDechirpBenchmarkROCm` (3 методов)
- `drv_gpu_lib::IBackend` (1 методов)
