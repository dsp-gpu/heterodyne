# 🤖 CLAUDE — `heterodyne`

> NCO, Mix (Down/Up), LFM Dechirp.
> Зависит от: `core` + `spectrum` + `signal_generators`. Глобальные правила → `../CLAUDE.md` + `.claude/rules/*.md`.

## 🎯 Что здесь

| Класс | Что делает |
|-------|-----------|
| `NCO` | Numerically Controlled Oscillator (phase accumulator) |
| `MixDownOp` / `MixUpOp` | Комплексное умножение с гетеродином |
| `HeterodyneDechirp` | Facade: LFM dechirp (mix с комплексно-сопряжённым chirp) |

## 📁 Структура

```
heterodyne/
├── include/dsp/heterodyne/
│   ├── heterodyne_dechirp.hpp      # facade
│   ├── gpu_context.hpp
│   ├── operations/                  # NCOOp, MixDownOp, MixUpOp, DechirpOp
│   └── strategies/
├── src/
├── kernels/rocm/                    # nco.hip, mix.hip, dechirp.hip
├── tests/
└── python/dsp_heterodyne_module.cpp
```

## ⚠️ Специфика

- **Phase unwrapping** не делать в MixDown — только в пост-процессе (если нужно).
- **Complex multiply**: `(a + bi)(c + di) = (ac - bd) + (ad + bc)i` — не использовать `std::complex<float>` в HIP kernel, работать покомпонентно.
- **Dechirp**: mix с `exp(-j·2π(f0·t + 0.5·k·t²))` — фазу считать в `double`, результат в `float`.
- **NCO table vs direct**: для короткой длины — direct `sincosf`, для длинной — таблица + интерполяция.

## 🚫 Запреты

- Не использовать `hipComplex` / `std::complex` в kernel — всё через `float2` / `__half2`.
- Не объединять Dechirp + FFT в один kernel — хранить раздельно для переиспользования.
- Не изобретать LFM-генератор — он в `signal_generators::LFMGenerator`.

## 🔗 Правила (path-scoped автоматически)

- `09-rocm-only.md`
- `05-architecture-ref03.md`
- `14-cpp-style.md` + `15-cpp-testing.md`
- `11-python-bindings.md`
