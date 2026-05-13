# Архитектурные паттерны репо `heterodyne`

> **Источник истины:** `heterodyne/.rag/_RAG.md` (теги `#pattern:Type:Class`, auto-inferred RAG_CLAUDE_C4 от 9.05).
> Brief'ы — из `key_classes:` того же манифеста (fallback из `rag_dsp.symbols`).
>
> Используется как источник для `dataset_v4` (collect_doc_deep подхватит Doc/Patterns.md).
> Alex: проверить + добавить руками то что не размечено в `_RAG.md tags:`.

## Facade

> Тонкий публичный API над набором операций. Стабильный → Python-биндинги не ломаются.


- **`dsp::heterodyne::HeterodyneDechirp`** — `heterodyne/include/heterodyne/heterodyne_dechirp.hpp:42`
  - Facade репо `heterodyne`: LFM dechirp (mix принятого сигнала с комплексно-сопряжённым chirp-эталоном). Pipeline: NCO → MixDown → LPF. Используется для сжатия LFM-импульса в PD-радаре. Стабильный публичный API → Python-биндинги не ломаются.
- **`dsp::heterodyne::HeterodyneProcessorROCm`** — `heterodyne/include/heterodyne/processors/heterodyne_processor_rocm.hpp:41`
  - Реализация IHeterodyneProcessor: запускает HIP-ядра dechirp_multiply (s_dc = conj(s_rx · s_ref)) и dechirp_correct (сдвиг f_beat → DC) через hiprtc-скомпилированный модуль. Управляет кэшем GPU-буферов (rx, ref, dc, corr, freq) с re-allocati

## Pipeline

> Композиция операций в цепочку. Конфиг → Pipeline объект.


- **`dsp::heterodyne::HeterodyneProcessorROCm`** — `heterodyne/include/heterodyne/processors/heterodyne_processor_rocm.hpp:41`
  - Реализация IHeterodyneProcessor: запускает HIP-ядра dechirp_multiply (s_dc = conj(s_rx · s_ref)) и dechirp_correct (сдвиг f_beat → DC) через hiprtc-скомпилированный модуль. Управляет кэшем GPU-буферов (rx, ref, dc, corr, freq) с re-allocati
- **`dsp::heterodyne::IHeterodyneProcessor`** — `heterodyne/include/heterodyne/i_heterodyne_processor.hpp:18`
  - Pure-virtual интерфейс для GPU-реализаций dechirp-операций: Dechirp (s_dc = s_rx · conj(s_tx)), Correct (сдвиг f_beat → DC), DechirpFromGPU (внешний GPU-буфер), DechirpWithGPURef (оба входа уже на GPU). Реализуется HeterodyneProcessorROCm (

## Strategy

> Семейство взаимозаменяемых алгоритмов за общим интерфейсом (`IPipelineStep`).


- **`dsp::heterodyne::IHeterodyneProcessor`** — `heterodyne/include/heterodyne/i_heterodyne_processor.hpp:18`
  - Pure-virtual интерфейс для GPU-реализаций dechirp-операций: Dechirp (s_dc = s_rx · conj(s_tx)), Correct (сдвиг f_beat → DC), DechirpFromGPU (внешний GPU-буфер), DechirpWithGPURef (оба входа уже на GPU). Реализуется HeterodyneProcessorROCm (


## См. также

- `heterodyne/.rag/arch/C2_container.md`
- `heterodyne/.rag/arch/C3_component.md`
- `heterodyne/.rag/arch/C4_code.md`
- `MemoryBank/.architecture/DSP-GPU_Design_C4_Full.md`

---

*Сгенерировано из `_RAG.md` тегов. Alex редактирует руками + коммитит.*
