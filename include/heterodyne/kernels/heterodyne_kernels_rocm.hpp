#pragma once

/**
 * @brief HIP kernel-source для heterodyne dechirp (multiply + correct).
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source для hiprtc.
 *       Один compilation unit содержит два ядра:
 *         - dechirp_multiply: dc = conj(rx * ref), даёт ПОЛОЖИТЕЛЬНУЮ
 *           f_beat = +μ·τ (умножение на conj(s_tx) с последующим conj
 *           эквивалентно conj(rx)·s_tx).
 *         - dechirp_correct:  output = input · exp(j·phase_step·n) —
 *           сдвиг тонала к 0 Гц (DC) для последующей фильтрации.
 * @note Layout: [num_antennas × num_samples] flat, channel-sequential.
 *       Grid: ((num_samples+255)/256, num_antennas), Block: 256.
 * @note Оптимизации (порт из OpenCL → HIP):
 *         OPT-5:  2D grid (x=sample, y=antenna) — нет div/mod внутри ядра.
 *         OPT-6:  phase_step[ant] = -2π·f_beat[ant]/fs предсчитан на CPU.
 *         OPT-7:  __launch_bounds__(256) — optimal RDNA register allocation.
 *         OPT-8:  aligned(8) float2_t — 64-bit load/store вместо двух 32-bit.
 *         OPT-9:  sincosf — один SFU-проход вместо cosf+sinf.
 *         OPT-10: Один hiprtc unit на оба ядра — экономия compile-time.
 *
 * История:
 *   - Создан:  2026-02-23 (порт dechirp_multiply.cl + dechirp_correct.cl на HIP)
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#if ENABLE_ROCM

namespace drv_gpu_lib {
namespace kernels {

// ════════════════════════════════════════════════════════════════════════════
// Combined kernel source: dechirp_multiply + dechirp_correct
// ════════════════════════════════════════════════════════════════════════════

inline const char* GetHeterodyneKernelSource_rocm() {
  return R"HIP(

// float2_t — complex number (hiprtc has no built-in float2)
// aligned(8) ensures 64-bit load/store instead of two 32-bit transactions
struct __attribute__((aligned(8))) float2_t {
    float x;
    float y;
};

// ════════════════════════════════════════════════════════════════════════════
// dechirp_multiply: output[ant][n] = conj(rx[ant][n] * ref[n])
//
// OPT-5: 2D grid (x=sample, y=antenna) — no div/mod
//   Grid: ((num_samples+255)/256, num_antennas), Block: (256)
//
// ref = conj(s_tx) from LfmConjugateGenerator (broadcast to all antennas).
// conj(rx * ref) = conj(s_rx * conj(s_tx)) = conj(s_rx) * s_tx
// This produces POSITIVE beat frequency: f_beat = +mu*tau
//
// Math: conj((a+jb)(c+jd)) = (ac-bd) - j(ad+bc)
// ════════════════════════════════════════════════════════════════════════════

extern "C" __launch_bounds__(256)
__global__ void dechirp_multiply(
    const float2_t* __restrict__ rx,
    const float2_t* __restrict__ ref,
          float2_t* __restrict__ dc_out,
    const int num_samples,
    const int num_antennas)
{
    const int n   = blockIdx.x * blockDim.x + threadIdx.x;
    const int ant = blockIdx.y;
    if (n >= num_samples || ant >= num_antennas) return;

    const int gid = ant * num_samples + n;

    const float2_t rx_v = rx[gid];
    const float2_t re_v = ref[n];          // broadcast: one ref for all antennas

    // conj(rx * ref): gives positive beat frequency
    float2_t out;
    out.x =  rx_v.x * re_v.x - rx_v.y * re_v.y;   // Re:  a*c - b*d
    out.y = -rx_v.x * re_v.y - rx_v.y * re_v.x;    // Im: -(a*d + b*c)
    dc_out[gid] = out;
}

// ════════════════════════════════════════════════════════════════════════════
// dechirp_correct: output = input * exp(j * phase_step * n)
//
// After dechirp, signal has a tone at f_beat.
// Multiplying by exp(j * phase_step * n) shifts spectrum to DC (0 Hz).
//
// OPT-5: 2D grid (x=sample, y=antenna) — no div/mod
// OPT-6: phase_step[] precomputed on CPU = -2*pi*f_beat/sample_rate
// OPT-9: sincosf computes sin+cos in single SFU pass
// ════════════════════════════════════════════════════════════════════════════

extern "C" __launch_bounds__(256)
__global__ void dechirp_correct(
    const float2_t* __restrict__ dc_in,
          float2_t* __restrict__ corrected,
    const float*    __restrict__ phase_step,
    const int num_samples,
    const int num_antennas)
{
    const int n   = blockIdx.x * blockDim.x + threadIdx.x;
    const int ant = blockIdx.y;
    if (n >= num_samples || ant >= num_antennas) return;

    const int gid = ant * num_samples + n;

    // OPT-6: phase = phase_step[ant] * n (no division in kernel)
    const float phase = phase_step[ant] * (float)n;

    // OPT-9: sincosf — один вызов SFU вместо двух раздельных cosf/sinf
    float sin_p, cos_p;
    sincosf(phase, &sin_p, &cos_p);

    // Complex multiply: corrected = dc_in * exp(j*phase)
    const float2_t in = dc_in[gid];
    float2_t out;
    out.x = in.x * cos_p - in.y * sin_p;
    out.y = in.y * cos_p + in.x * sin_p;
    corrected[gid] = out;
}

)HIP";
}

}  // namespace kernels
}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
