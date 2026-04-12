// SPDX-License-Identifier: Apache-2.0
//
// Self-contained neural-texture forward-inference kernel for the Slang
// autotuning benchmark. The activation function is exposed as a single
// [Tunable] extern struct so the Python side can enumerate impls via
// TuningSpace.discover() and specialize per trial using
// FunctionNode.with_settings().
//
// Width (W), depth (D), and frequency-encoding band count (FB) are fixed at
// module-load time via preprocessor defines. The Python driver reloads the
// module per (W, D, FB) outer-loop point; activation is the inner tunable
// axis, keyed through the slangpy CallDataCache on the link bindings.

import slangpy;

#ifndef W
#define W 32
#endif
#ifndef D
#define D 2
#endif
#ifndef FB
#define FB 6
#endif

static const int kWidth = W;
static const int kDepth = D;
static const int kFreqBands = FB;
// Frequency encoding: raw xy (2) + {sin,cos} * {x,y} * FB bands
static const int kInputDim = 2 + 4 * FB;

// --------------------------------------------------------------------------- //
// Tunable activation
// --------------------------------------------------------------------------- //

public interface IBenchAct
{
    float apply(float x);
}

public struct TReLU : IBenchAct
{
    public float apply(float x) { return max(x, 0.0f); }
}

public struct TLeakyReLU : IBenchAct
{
    public float apply(float x) { return x >= 0.0f ? x : 0.01f * x; }
}

public struct TELU : IBenchAct
{
    public float apply(float x) { return x >= 0.0f ? x : (exp(x) - 1.0f); }
}

public struct TSmeLU : IBenchAct
{
    // Smooth ReLU: quadratic in a small transition band around 0.
    public float apply(float x)
    {
        const float beta = 0.5f;
        if (x <= -beta) return 0.0f;
        if (x >= beta) return x;
        float t = x + beta;
        return (t * t) / (4.0f * beta);
    }
}

public struct TSwish : IBenchAct
{
    public float apply(float x) { return x / (1.0f + exp(-x)); }
}

public struct TTanh : IBenchAct
{
    public float apply(float x) { return tanh(x); }
}

public struct TSigmoid : IBenchAct
{
    public float apply(float x) { return 1.0f / (1.0f + exp(-x)); }
}

[Tunable] public extern struct Act : IBenchAct = TReLU;

// --------------------------------------------------------------------------- //
// Forward MLP
// --------------------------------------------------------------------------- //

// Parameter layout (flat float buffer, row-major):
//   layer 0 (input -> hidden):
//     W0: kInputDim * kWidth
//     b0: kWidth
//   layers 1..kDepth-1 (hidden -> hidden):
//     Wi: kWidth * kWidth
//     bi: kWidth
//   output (hidden -> 3):
//     Wout: kWidth * 3
//     bout: 3
//
// The flat layout avoids having to name N distinct buffers from the Python
// side and keeps the slangpy binding surface minimal.

float3 forwardMLP(float2 uv, StructuredBuffer<float> params)
{
    // Frequency encoding: [x, y, sin(f*x), cos(f*x), sin(f*y), cos(f*y), ...]
    float enc[kInputDim];
    enc[0] = uv.x;
    enc[1] = uv.y;
    [ForceUnroll]
    for (int i = 0; i < kFreqBands; ++i)
    {
        float f = (float)(1 << i) * 3.14159265f;
        enc[2 + 4 * i + 0] = sin(f * uv.x);
        enc[2 + 4 * i + 1] = cos(f * uv.x);
        enc[2 + 4 * i + 2] = sin(f * uv.y);
        enc[2 + 4 * i + 3] = cos(f * uv.y);
    }

    int off = 0;

    // Layer 0: kInputDim -> kWidth
    float h[kWidth];
    [ForceUnroll]
    for (int r = 0; r < kWidth; ++r)
    {
        float s = params[off + kInputDim * kWidth + r]; // bias
        [ForceUnroll]
        for (int c = 0; c < kInputDim; ++c)
        {
            s += params[off + r * kInputDim + c] * enc[c];
        }
        h[r] = (Act()).apply(s);
    }
    off += kInputDim * kWidth + kWidth;

    // Hidden layers: kWidth -> kWidth
    for (int l = 1; l < kDepth; ++l)
    {
        float hn[kWidth];
        [ForceUnroll]
        for (int r = 0; r < kWidth; ++r)
        {
            float s = params[off + kWidth * kWidth + r]; // bias
            [ForceUnroll]
            for (int c = 0; c < kWidth; ++c)
            {
                s += params[off + r * kWidth + c] * h[c];
            }
            hn[r] = (Act()).apply(s);
        }
        [ForceUnroll]
        for (int r = 0; r < kWidth; ++r)
            h[r] = hn[r];
        off += kWidth * kWidth + kWidth;
    }

    // Output: kWidth -> 3 (no activation; exp at the end mimics bench.py)
    float3 y;
    [ForceUnroll]
    for (int r = 0; r < 3; ++r)
    {
        float s = params[off + kWidth * 3 + r]; // bias
        [ForceUnroll]
        for (int c = 0; c < kWidth; ++c)
        {
            s += params[off + r * kWidth + c] * h[c];
        }
        y[r] = exp(s);
    }

    return y;
}

// slangpy entry point — one call per (uv) sample, vectorised per pixel.
public float3 evalPixel(float2 uv, StructuredBuffer<float> params)
{
    return forwardMLP(uv, params);
}
