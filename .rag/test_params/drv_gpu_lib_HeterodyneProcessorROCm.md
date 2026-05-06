---
schema_version: 1
repo: heterodyne
class_fqn: drv_gpu_lib::HeterodyneProcessorROCm
file: E:/DSP-GPU/heterodyne/include/heterodyne/processors/heterodyne_processor_rocm.hpp
line: 41
brief: "Обрабатывает ЛЧМ-сигналы через ROCm/HIP для радиолокационной дешифровки"
methods_total: 12
methods_with_doxygen: 12
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['ROCm обработка', 'ЛЧМ дешифровка', 'GPU радиолокация', 'stretch-processing', 'частота биений']
synonyms_en: ['ROCm processing', 'chirp dechirping', 'GPU radar', 'stretch processing', 'beat frequency']
tags: ['ROCm', 'GPU', 'дешифровка', 'ЛЧМ', 'обработка сигналов']
---

# `drv_gpu_lib::HeterodyneProcessorROCm` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo heterodyne --class HeterodyneProcessorROCm`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=heterodyne__heterodyne_processor_rocm__class_overview__v1 -->

**ЧТО**: Обрабатывает ЛЧМ-сигналы через ROCm/HIP для радиолокационной дешифровки

**ЗАЧЕМ**: Реализует stretch-processing для определения дальности и частоты биений на AMD GPU

**КАК**: Использует HIP для GPU-вычислений, поддерживает профилирование, предоставляет CPU/GPU версии методов

**Пример**:
```cpp
#include "heterodyne/processors/heterodyne_processor_rocm.hpp"
using namespace drv_gpu_lib;

int main() {
    auto backend = std::make_unique<ROCmBackend>();
    HeterodyneProcessorROCm processor{backend.get()};

    std::vector<std::complex<float>> rx_data = ...;
    std::vector<std::complex<float>> ref_data = ...;
    HeterodyneParams params = ...;

    auto result = processor.Dechirp(rx_data, ref_data, params);
    // или с профилированием: HeterodyneROCmProfEvents events;
    // auto result = processor.Dechirp(rx_data, ref_data, params, &events);
}
```

<!-- /rag-block -->

## Связанные секции из Doc/

- `heterodyne__api__s_6_heterodyneprocessorrocm__v1` (s_6_heterodyneprocessorrocm): ## 6. HeterodyneProcessorROCm  **Файл**: `include/processors/heterodyne_processor_rocm.hpp` **Требует**: `ENABLE_ROCM=1` (Linux + AMD GPU). При `!ENABLE_ROCM` — stub, все методы бросают `std::runtime_…
- `heterodyne__gpu__overview__v1` (overview): ## 1. Обзор и назначение  HeterodyneDechirp — процессор **stretch-processing** для ЛЧМ-радара. Принимает комплексный сигнал с антенн, умножает на сопряжённый опорный ЛЧМ, находит частоту биений и вычи…
- `heterodyne__api__section_001__v1` (section): ## Содержание  1. [HeterodyneDechirp (Facade)](#1-heterodynedechirp) 2. [HeterodyneParams](#2-heterodyneparams) 3. [HeterodyneResult / AntennaDechirpResult](#3-heterodyneresult) 4. [IHeterodyneProcess…

## Public-методы (12)

## Method 1: `Dechirp`

**Сигнатура** (`heterodyne_processor_rocm.hpp:107`):
```cpp
std::vector<std::complex<float>> Dechirp( const std::vector<std::complex<float>>& rx_data, const std::vector<std::complex<float>>& ref_data, const HeterodyneParams& params) override { return Dechirp(rx_data, ref_data, params, nullptr);
```

**Параметры**:
- `rx_data` — `const std::vector<std::complex<float>>&`
- `ref_data` — `const std::vector<std::complex<float>>&`
- `params` — `const HeterodyneParams&`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
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
```

## Method 2: `Dechirp`

**Сигнатура** (`heterodyne_processor_rocm.hpp:127`):
```cpp
std::vector<std::complex<float>> Dechirp( const std::vector<std::complex<float>>& rx_data, const std::vector<std::complex<float>>& ref_data, const HeterodyneParams& params, HeterodyneROCmProfEvents* prof_events)
```

**Параметры**:
- `rx_data` — `const std::vector<std::complex<float>>&`
- `ref_data` — `const std::vector<std::complex<float>>&`
- `params` — `const HeterodyneParams&`
- `prof_events` — `HeterodyneROCmProfEvents*` *(pointer)*

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Dechirp с CPU-входом и сборкой ROCm-событий профилирования. H2D → multiply kernel → D2H.
   *
   * @param rx_data CPU-данные [num_antennas × num_samples] complex<float> (flat).
   * @param ref_data Reference [num_samples] complex<float> = conj(s_tx).
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return Dechirp-данные [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
```

## Method 3: `Correct`

**Сигнатура** (`heterodyne_processor_rocm.hpp:145`):
```cpp
std::vector<std::complex<float>> Correct( const std::vector<std::complex<float>>& dc_data, const std::vector<float>& f_beat_hz, const HeterodyneParams& params) override { return Correct(dc_data, f_beat_hz, params, nullptr);
```

**Параметры**:
- `dc_data` — `const std::vector<std::complex<float>>&`
- `f_beat_hz` — `const std::vector<float>&`
- `params` — `const HeterodyneParams&`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
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
```

## Method 4: `Correct`

**Сигнатура** (`heterodyne_processor_rocm.hpp:165`):
```cpp
std::vector<std::complex<float>> Correct( const std::vector<std::complex<float>>& dc_data, const std::vector<float>& f_beat_hz, const HeterodyneParams& params, HeterodyneROCmProfEvents* prof_events)
```

**Параметры**:
- `dc_data` — `const std::vector<std::complex<float>>&`
- `f_beat_hz` — `const std::vector<float>&`
- `params` — `const HeterodyneParams&`
- `prof_events` — `HeterodyneROCmProfEvents*` *(pointer)*

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Частотная коррекция (сдвиг f_beat → DC) с ROCm-профилированием. exp(-j·2π·f_beat·t).
   *
   * @param dc_data Dechirp-данные [num_antennas × num_samples] complex<float>.
   * @param f_beat_hz Beat-частоты по антеннам [num_antennas], Гц.
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return Скорректированные данные [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
```

## Method 5: `DechirpFromGPU`

**Сигнатура** (`heterodyne_processor_rocm.hpp:184`):
```cpp
std::vector<std::complex<float>> DechirpFromGPU( void* rx_gpu_ptr, const std::vector<std::complex<float>>& ref_data, const HeterodyneParams& params) override { return DechirpFromGPU(rx_gpu_ptr, ref_data, params, nullptr);
```

**Параметры**:
- `rx_gpu_ptr` — `void*` *(pointer)* *(void\*)*
- `ref_data` — `const std::vector<std::complex<float>>&`
- `params` — `const HeterodyneParams&`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Перегрузка `DechirpFromGPU` — wrapper, делегирует в основную с аргументами (rx_gpu_ptr, ref_data, params, nullptr). Dechirp из внешнего GPU-буфера (void* = hipDeviceptr_t).
   * @see DechirpFromGPU (основная перегрузка)
   *
   * @param rx_gpu_ptr Внешний GPU-буфер (hipDeviceptr_t) [num_antennas × num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param ref_data Reference [num_samples] complex<float> на CPU (H2D внутри метода).
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   *
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
```

## Method 6: `DechirpFromGPU`

**Сигнатура** (`heterodyne_processor_rocm.hpp:205`):
```cpp
std::vector<std::complex<float>> DechirpFromGPU( void* rx_gpu_ptr, const std::vector<std::complex<float>>& ref_data, const HeterodyneParams& params, HeterodyneROCmProfEvents* prof_events)
```

**Параметры**:
- `rx_gpu_ptr` — `void*` *(pointer)* *(void\*)*
- `ref_data` — `const std::vector<std::complex<float>>&`
- `params` — `const HeterodyneParams&`
- `prof_events` — `HeterodyneROCmProfEvents*` *(pointer)*

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Dechirp с внешним GPU-входом и ROCm-профилированием. Без H2D для rx, есть H2D для ref.
   *
   * @param rx_gpu_ptr Внешний GPU-буфер (hipDeviceptr_t) [num_antennas × num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param ref_data Reference [num_samples] complex<float> на CPU (H2D внутри метода).
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
```

## Method 7: `DechirpWithGPURef`

**Сигнатура** (`heterodyne_processor_rocm.hpp:225`):
```cpp
std::vector<std::complex<float>> DechirpWithGPURef( void* rx_gpu_ptr, void* ref_gpu_ptr, const HeterodyneParams& params) override { return DechirpWithGPURef(rx_gpu_ptr, ref_gpu_ptr, params, nullptr);
```

**Параметры**:
- `rx_gpu_ptr` — `void*` *(pointer)* *(void\*)*
- `ref_gpu_ptr` — `void*` *(pointer)* *(void\*)*
- `params` — `const HeterodyneParams&`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief Перегрузка `DechirpWithGPURef` — wrapper, делегирует в основную с аргументами (rx_gpu_ptr, ref_gpu_ptr, params, nullptr). OPT-3: и rx, и ref уже на GPU.
   * @see DechirpWithGPURef (основная перегрузка)
   *
   * @param rx_gpu_ptr Внешний GPU-буфер с rx [num_antennas × num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param ref_gpu_ptr Внешний GPU-буфер с conj(LFM) ref [num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   *
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
```

## Method 8: `DechirpWithGPURef`

**Сигнатура** (`heterodyne_processor_rocm.hpp:247`):
```cpp
std::vector<std::complex<float>> DechirpWithGPURef( void* rx_gpu_ptr, void* ref_gpu_ptr, const HeterodyneParams& params, HeterodyneROCmProfEvents* prof_events)
```

**Параметры**:
- `rx_gpu_ptr` — `void*` *(pointer)* *(void\*)*
- `ref_gpu_ptr` — `void*` *(pointer)* *(void\*)*
- `params` — `const HeterodyneParams&`
- `prof_events` — `HeterodyneROCmProfEvents*` *(pointer)*

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
/**
   * @brief OPT-3 dechirp: rx и ref уже на GPU, без PCIe для ref. С ROCm-профилированием.
   *
   * @param rx_gpu_ptr Внешний GPU-буфер с rx [num_antennas × num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param ref_gpu_ptr Внешний GPU-буфер с conj(LFM) ref [num_samples]; caller владеет.
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param params Параметры LFM (num_antennas, num_samples, sample_rate, ...).
   *   @test_ref HeterodyneParams
   * @param prof_events Сборщик ROCm-событий профилирования (опционально).
   *   @test { values=[nullptr], error_values=[0xDEADBEEF, null] }
   *
   * @return Dechirp-данные на CPU [num_antennas × num_samples] complex<float>.
   *   @test_check result.size() == params.num_antennas * params.num_samples
   */
```

## Method 9: `Dechirp`

**Сигнатура** (`heterodyne_processor_rocm.hpp:306`):
```cpp
std::vector<std::complex<float>> Dechirp( const std::vector<std::complex<float>>& /*rx_data*/, const std::vector<std::complex<float>>& /*ref_data*/, const HeterodyneParams& /*params*/) override { throw std::runtime_error("HeterodyneProcessorROCm::Dechirp: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const std::vector<std::complex<float>>&`
- `_unnamed_` — `const std::vector<std::complex<float>>&`
- `_unnamed_` — `const HeterodyneParams&`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
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
```

## Method 10: `Correct`

**Сигнатура** (`heterodyne_processor_rocm.hpp:323`):
```cpp
std::vector<std::complex<float>> Correct( const std::vector<std::complex<float>>& /*dc_data*/, const std::vector<float>& /*f_beat_hz*/, const HeterodyneParams& /*params*/) override { throw std::runtime_error("HeterodyneProcessorROCm::Correct: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const std::vector<std::complex<float>>&`
- `_unnamed_` — `const std::vector<float>&`
- `_unnamed_` — `const HeterodyneParams&`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
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
```

## Method 11: `DechirpFromGPU`

**Сигнатура** (`heterodyne_processor_rocm.hpp:340`):
```cpp
std::vector<std::complex<float>> DechirpFromGPU( void* /*rx_gpu_ptr*/, const std::vector<std::complex<float>>& /*ref_data*/, const HeterodyneParams& /*params*/) override { throw std::runtime_error("HeterodyneProcessorROCm::DechirpFromGPU: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `void*` *(pointer)* *(void\*)*
- `_unnamed_` — `const std::vector<std::complex<float>>&`
- `_unnamed_` — `const HeterodyneParams&`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
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
```

## Method 12: `DechirpWithGPURef`

**Сигнатура** (`heterodyne_processor_rocm.hpp:357`):
```cpp
std::vector<std::complex<float>> DechirpWithGPURef( void* /*rx_gpu_ptr*/, void* /*ref_gpu_ptr*/, const HeterodyneParams& /*params*/) override { throw std::runtime_error("HeterodyneProcessorROCm::DechirpWithGPURef: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `void*` *(pointer)* *(void\*)*
- `_unnamed_` — `void*` *(pointer)* *(void\*)*
- `_unnamed_` — `const HeterodyneParams&`

**Возвращает**: `std::vector<std::complex<float>>`

**Doxygen-источник**:
```cpp
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
```

