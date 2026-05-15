// SPDX-License-Identifier: Apache-2.0
//
// Source-specialized neural-texture kernel for the OpenTuner showcase.
//
// This file deliberately does not use [Tunable], link-time specialization, or
// Slang reflection. An OpenTuner-style driver still mirrors the full search
// space in Python, substitutes the placeholders below, and compiles a concrete
// Slang module for each proposed configuration.

import slangpy;
import activation;

static const int kWidth = __K_WIDTH__;
static const int kDepth = __K_DEPTH__;
static const int kFreqBands = __K_FREQ_BANDS__;

// Frequency encoding: raw xy (2) + {sin,cos} * {x,y} * FB bands
static const int kInputDim = 2 + 4 * kFreqBands;

float3 forwardMLP(float2 uv, StructuredBuffer<float> params)
{
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

    float h[kWidth];
    [ForceUnroll]
    for (int r = 0; r < kWidth; ++r)
    {
        float s = params[off + kInputDim * kWidth + r];
        [ForceUnroll]
        for (int c = 0; c < kInputDim; ++c)
        {
            s += params[off + r * kInputDim + c] * enc[c];
        }
        h[r] = (__ACTIVATION__()).apply(s);
    }
    off += kInputDim * kWidth + kWidth;

    for (int l = 1; l < kDepth; ++l)
    {
        float hn[kWidth];
        [ForceUnroll]
        for (int r = 0; r < kWidth; ++r)
        {
            float s = params[off + kWidth * kWidth + r];
            [ForceUnroll]
            for (int c = 0; c < kWidth; ++c)
            {
                s += params[off + r * kWidth + c] * h[c];
            }
            hn[r] = (__ACTIVATION__()).apply(s);
        }
        [ForceUnroll]
        for (int r = 0; r < kWidth; ++r)
            h[r] = hn[r];
        off += kWidth * kWidth + kWidth;
    }

    float3 y;
    [ForceUnroll]
    for (int r = 0; r < 3; ++r)
    {
        float s = params[off + kWidth * 3 + r];
        [ForceUnroll]
        for (int c = 0; c < kWidth; ++c)
        {
            s += params[off + r * kWidth + c] * h[c];
        }
        y[r] = exp(s);
    }

    return y;
}

public float3 evalPixel(float2 uv, StructuredBuffer<float> params)
{
    return forwardMLP(uv, params);
}
