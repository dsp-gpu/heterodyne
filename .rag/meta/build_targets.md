<!-- type:meta_targets repo:heterodyne source:heterodyne/CMakeLists.txt -->

# Build Targets — heterodyne

## Targets

- **`DspHeterodyne`** (library)
  - PUBLIC: `DspCore::DspCore`, `DspSpectrum::DspSpectrum`, `DspSignalGenerators::DspSignalGenerators`

## BUILD-флаги (option)

- `DSP_HETERODYNE_BUILD_TESTS` (default `ON`) — Build tests
- `DSP_HETERODYNE_BUILD_PYTHON` (default `OFF`) — Build Python bindings

## Зависимости от DSP репо

- `core` — через `fetch_dsp_core()`
- `signal_generators` — через `fetch_dsp_signal_generators()`
- `spectrum` — через `fetch_dsp_spectrum()`

## External find_package

- `hip` (required)
- `hipfft` (required)
