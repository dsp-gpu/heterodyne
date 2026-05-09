---
schema_version: 1
repo: heterodyne
arch_level: c2
tags:
  - "#level:c2"
  - "#repo:heterodyne"
  - "#layer:compute"
  - "#namespace:drv_gpu_lib"
  - "#namespace:test_heterodyne_rocm"
description: "C2 Container — namespace tree и зависимости репо heterodyne."
---

# C2 Container — `heterodyne` (layer=compute)

## Namespaces (top по числу классов)

- `drv_gpu_lib`
- `test_heterodyne_rocm`

## Public modules (`include/heterodyne/`)

- `kernels/`
- `processors/`

## Зависимости (depends_on)

`core`

## Используется (used_by)

`DSP`

## Top key_classes

| Class | Namespace | Methods | TestParams |
|-------|-----------|--------:|-----------:|
| `HeterodyneProcessorROCm` | `drv_gpu_lib` | 24 | 42 |
| `HeterodyneDechirp` | `drv_gpu_lib` | 8 | 5 |
| `HeterodyneResult` | `drv_gpu_lib` | 1 | 5 |
| `IHeterodyneProcessor` | `drv_gpu_lib` | 2 | 1 |
| `PyHeterodyneDechirp` | `(global)` | 6 | 0 |
