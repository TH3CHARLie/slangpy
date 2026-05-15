// SPDX-License-Identifier: Apache-2.0
//
// Neural-texture forward-inference kernel for the Slang autotuning benchmark.
//
// Tunable axes are split across modules:
//   - Activation function is imported from activation.slang.
//   - Width, depth, and frequency bands are declared in this entrypoint module.
//
// The Python driver discovers the full choice space via reflection and
// specializes each variant through module linking — no preprocessor defines.

import slangpy;
import activation;

[Tunable(16, 32, 64, 128)]
public extern static const int kWidth = 32;

[Tunable(2, 3, 4)]
public extern static const int kDepth = 2;

[Tunable(6)]
public extern static const int kFreqBands = 6;

// Frequency encoding: raw xy (2) + {sin,cos} * {x,y} * FB bands
static const int kInputDim = 2 + 4 * kFreqBands;

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
