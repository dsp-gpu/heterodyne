<!-- type:meta_cmake_specific repo:heterodyne inherits:dsp_gpu__root__meta_cmake_common__v1 -->

# CMake Specific — heterodyne

```yaml
inherits: dsp_gpu__root__meta_cmake_common__v1
specific_only: true
target: DspHeterodyne
description: "LFM Dechirp, NCO, MixDown/MixUp"
adds_find_package: [hip, hipfft]
adds_links: [DspCore::DspCore, DspSignalGenerators::DspSignalGenerators, DspSpectrum::DspSpectrum]
```

## Project

- **Target**: `DspHeterodyne`
- **Описание**: LFM Dechirp, NCO, MixDown/MixUp

## Уникальные find_package

```cmake
find_package(hip REQUIRED)
find_package(hipfft REQUIRED)
```

## Линкуемые библиотеки

```cmake
target_link_libraries(DspHeterodyne PUBLIC
  DspCore::DspCore
  DspSignalGenerators::DspSignalGenerators
  DspSpectrum::DspSpectrum
)
```

## Исходники (2 файлов)

```cmake
target_sources(DspHeterodyne PRIVATE
  src/heterodyne/heterodyne_dechirp.cpp
  src/heterodyne/heterodyne_processor_rocm.cpp
)
```

## Прочие специфичные строки (9)

```cmake
DESCRIPTION "LFM Dechirp, NCO, MixDown/MixUp"
PUBLIC <TARGET>::<TARGET> <TARGET>::<TARGET> <TARGET>::<TARGET>
fetch_dsp_signal_generators()
fetch_dsp_spectrum()
find_package(hip    REQUIRED)
find_package(hipfft REQUIRED)
src/heterodyne/heterodyne_dechirp.cpp
src/heterodyne/heterodyne_processor_rocm.cpp
target_link_libraries(<TARGET>
```

