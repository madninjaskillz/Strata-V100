// native_route_fused against native_router_top10_multi + native_route_resident + doorbell_publish_res,
// bit for bit on every output (ids, weights, the published ids / rows, the ring, the counters), and each one's time
// inside a captured graph.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

int main() {
    const int NE = 512, KK = 10, N = 2560, TM = 8;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    float *logits, *w[2], *x;
    int32_t *ids[2], *res;
    unsigned long long* stats[2];
    cudaMalloc(&logits, TM * NE * 4); cudaMalloc(&x, TM * N * 4); cudaMalloc(&res, NE * 4);
    for (int i = 0; i < 2; ++i) { cudaMalloc(&w[i], TM * KK * 4); cudaMalloc(&ids[i], TM * KK * 4); cudaMalloc(&stats[i], 8 * 8); }
    float* xo[2]; int32_t* io[2]; uint32_t* seq[2];
    for (int i = 0; i < 2; ++i) {
        cudaHostAlloc(&xo[i], TM * N * 4, cudaHostAllocMapped);
        cudaHostAlloc(&io[i], TM * KK * 4, cudaHostAllocMapped);
        cudaHostAlloc(&seq[i], 64, cudaHostAllocMapped);
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
    std::vector<float> hl(TM * NE), hx(TM * N);
    std::vector<int32_t> hres(NE);
    int fails = 0, cases = 0;
    for (int trial = 0; trial < 42; ++trial) {
        for (auto& v : hl) v = 2.0f * nd(rng);
        if (trial % 5 == 0) for (int i = 0; i < TM * NE; i += 7) hl[i] = hl[(i + 13) % (TM * NE)];   // ties
        for (auto& v : hx) v = nd(rng);
        const int pct = trial % 4 == 0 ? 100 : (trial % 4 == 1 ? 97 : (trial % 4 == 2 ? 60 : 0));
        for (auto& v : hres) v = (int) (rng() % 100) < pct ? (int) (rng() % 4096) : -1;
        cudaMemcpy(logits, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(res, hres.data(), hres.size() * 4, cudaMemcpyHostToDevice);
        for (int T = 1; T <= TM; ++T)
            for (int swap = 0; swap < 2; ++swap)
                for (int topk = 0; topk < 2; ++topk)
                    for (int nores = 0; nores < 2; ++nores) {
                        if (swap && nores) continue;
                        const int32_t* dres = nores ? nullptr : res;
                        for (int i = 0; i < 2; ++i) {
                            cudaMemset(w[i], 0xee, TM * KK * 4); cudaMemset(ids[i], 0xee, TM * KK * 4);
                            cudaMemset(stats[i], 0, 64);
                            std::memset(xo[i], 0x55, TM * N * 4); std::memset(io[i], 0x55, TM * KK * 4);
                            seq[i][0] = 41u;
                            if (!topk) K::native_router_top10_multi(logits, ids[i], w[i], T, s);
                        }
                        // reference: the separate launches
                        if (topk) K::native_router_top10_multi(logits, ids[0], w[0], T, s);
                        if (swap) K::native_route_resident(logits, ids[0], w[0], res, T, 1.5f, 6, 9, stats[0], s);
                        K::doorbell_publish_res(x, ids[0], dres, NE, (int64_t) T * N, (int64_t) T * KK, xo[0], io[0], seq[0], s);
                        // fused
                        K::native_route_fused(logits, ids[1], w[1], T, topk != 0, swap ? res : nullptr, 1.5f, 6, 9,
                                              swap ? stats[1] : nullptr, x, dres, NE, (int64_t) T * N, xo[1], io[1], seq[1], s);
                        cudaStreamSynchronize(s);
                        std::vector<uint8_t> a(TM * KK * 8 + 64), b(a.size());
                        cudaMemcpy(a.data(), ids[0], TM * KK * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(a.data() + TM * KK * 4, w[0], TM * KK * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(a.data() + TM * KK * 8, stats[0], 64, cudaMemcpyDeviceToHost);
                        cudaMemcpy(b.data(), ids[1], TM * KK * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(b.data() + TM * KK * 4, w[1], TM * KK * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(b.data() + TM * KK * 8, stats[1], 64, cudaMemcpyDeviceToHost);
                        const bool ok = a == b && std::memcmp(xo[0], xo[1], TM * N * 4) == 0 &&
                                        std::memcmp(io[0], io[1], TM * KK * 4) == 0 && seq[0][0] == seq[1][0] && seq[0][0] == 42u;
                        ++cases;
                        if (!ok) {
                            ++fails;
                            if (fails <= 10)
                                std::printf("DIFFERS trial %d T %d swap %d topk %d nores %d (ids/w/stats %d, x_out %d, ids_out %d, ring %u %u)\n",
                                            trial, T, swap, topk, nores, (int) (a == b), std::memcmp(xo[0], xo[1], TM * N * 4) == 0,
                                            std::memcmp(io[0], io[1], TM * KK * 4) == 0, seq[0][0], seq[1][0]);
                        }
                    }
    }
    std::printf("route_fused_parity: %d cases, %d failures\n", cases, fails);
    // timing inside a graph: 48 "layers"
    cudaStream_t gs;
    cudaStreamCreateWithFlags(&gs, cudaStreamNonBlocking);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int T = 1; T <= 4; ++T)
        for (int f = 0; f < 5; ++f) {
            cudaGraph_t g; cudaGraphExec_t ge;
            cudaStreamBeginCapture(gs, cudaStreamCaptureModeGlobal);
            for (int l = 0; l < 48; ++l) {
                if (f == 0) {
                    K::native_router_top10_multi(logits, ids[0], w[0], T, gs);
                    K::native_route_resident(logits, ids[0], w[0], res, T, 1.5f, 6, 9, stats[0], gs);
                    K::doorbell_publish_res(x, ids[0], res, NE, (int64_t) T * N, (int64_t) T * KK, xo[0], io[0], seq[0], gs);
                } else if (f == 2) { K::native_router_top10_multi(logits, ids[0], w[0], T, gs);
                } else if (f == 3) { K::native_route_resident(logits, ids[0], w[0], res, T, 1.5f, 6, 9, stats[0], gs);
                } else if (f == 4) { K::doorbell_publish_res(x, ids[0], res, NE, (int64_t) T * N, (int64_t) T * KK, xo[0], io[0], seq[0], gs);
                } else {
                    K::native_route_fused(logits, ids[1], w[1], T, true, res, 1.5f, 6, 9, stats[1], x, res, NE, (int64_t) T * N,
                                          xo[1], io[1], seq[1], gs);
                }
            }
            cudaStreamEndCapture(gs, &g);
            cudaGraphInstantiate(&ge, g, 0);
            for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, gs);
            cudaStreamSynchronize(gs);
            float best = 1e30f;
            for (int r = 0; r < 5; ++r) {
                cudaEventRecord(e0, gs);
                for (int i = 0; i < 20; ++i) cudaGraphLaunch(ge, gs);
                cudaEventRecord(e1, gs); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) best = ms;
            }
            std::printf("T %d %s: %.2f us per layer\n", T, f == 0 ? "separate" : f == 1 ? "fused   " : f == 2 ? "top10 only" : f == 3 ? "resident only" : "doorbell only", 1e3 * best / (20 * 48));
            cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
        }
    return fails != 0;
}
