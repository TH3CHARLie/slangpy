// SPDX-License-Identifier: Apache-2.0
//
// Differentiable SDF raymarching: fit a neural SDF (tiny MLP) to a target
// analytic sphere by backpropagating per-pixel L2 loss through a sphere-traced
// raymarch loop.
//
// This variant exposes the autodiff checkpointing schedule with three
// link-time bool constants. `[CheckpointPreference(true)]` behaves like
// `[PreferCheckpoint]`; `[CheckpointPreference(false)]` behaves like
// `[PreferRecompute]`.
//
// Axes:
//   PreferCheckpointMLP    - schedule for MLP forward
//   PreferCheckpointSDF    - schedule for neural SDF eval wrapper
//   PreferCheckpointMarch  - schedule for the entire raymarch loop

import slangpy;

#ifndef N_STEPS
#define N_STEPS 32
#endif
#ifndef W
#define W 16
#endif

extern static const bool PreferCheckpointMLP = true;
extern static const bool PreferCheckpointSDF = true;
extern static const bool PreferCheckpointMarch = true;

static const int kSteps = N_STEPS;
static const int kWidth = W;
static const int kInput = 3;

// Parameter layout (flat):
//   [0 .. kInput*kWidth)                    - W0 (kWidth rows x kInput cols)
//   [kInput*kWidth .. kInput*kWidth+kWidth) - b0 (kWidth)
//   [... .. ... + kWidth)                   - W1 (1 row x kWidth cols)
//   [... .. ... + 1)                        - b1 (1)
static const int kParams = kInput * kWidth + kWidth + kWidth + 1;

// --------------------------------------------------------------------------- //
// MLP forward - schedule axis #1
// --------------------------------------------------------------------------- //

[Differentiable]
float mlp_body(no_diff float3 p, float params[kParams])
{
    float h[kWidth];
    [ForceUnroll]
    for (int r = 0; r < kWidth; ++r)
    {
        float s = params[kInput * kWidth + r]; // b0[r]
        s += params[r * kInput + 0] * p.x;
        s += params[r * kInput + 1] * p.y;
        s += params[r * kInput + 2] * p.z;
        h[r] = s >= 0.0 ? s : 0.01 * s; // Leaky ReLU
    }
    int off = kInput * kWidth + kWidth;
    float outv = params[off + kWidth]; // b1
    [ForceUnroll]
    for (int c = 0; c < kWidth; ++c)
        outv += params[off + c] * h[c];
    return outv;
}

[Differentiable]
[CheckpointPreference(PreferCheckpointMLP)]
float mlp_forward(no_diff float3 p, float params[kParams])
{
    return mlp_body(p, params);
}

// --------------------------------------------------------------------------- //
// Neural SDF eval wrapper - schedule axis #2
// --------------------------------------------------------------------------- //
//
// In principle this is just a pass-through around the MLP. Wrapping it lets
// us schedule the SDF eval independently of the MLP: useful when the MLP is
// cheap but the enclosing per-step state is expensive to keep live.

[Differentiable]
float sdf_body(no_diff float3 p, float params[kParams])
{
    return mlp_forward(p, params);
}

[Differentiable]
[CheckpointPreference(PreferCheckpointSDF)]
float sdf_eval(no_diff float3 p, float params[kParams])
{
    return sdf_body(p, params);
}

// --------------------------------------------------------------------------- //
// Raymarch loop - schedule axis #3
// --------------------------------------------------------------------------- //
//
// Fixed-iteration sphere-tracing loop. No early-out so the loop is uniform;
// this simplifies the autodiff state layout.

[Differentiable]
float march_body(no_diff float3 ro, no_diff float3 rd, float params[kParams])
{
    float t = 0.0;
    [MaxIters(kSteps)]
    for (int i = 0; i < kSteps; ++i)
    {
        float3 p = ro + rd * t;
        float d = sdf_eval(p, params);
        // Smooth-saturating step: tanh keeps |d| bounded without a hard
        // cutoff that zeros gradients at the boundary.
        t += 3.0 * tanh(d / 3.0);
    }
    return t;
}

[Differentiable]
[CheckpointPreference(PreferCheckpointMarch)]
float march(no_diff float3 ro, no_diff float3 rd, float params[kParams])
{
    return march_body(ro, rd, params);
}

// --------------------------------------------------------------------------- //
// Render + loss
// --------------------------------------------------------------------------- //
//
// Per-pixel: march a ray, convert marched distance to a soft hit weight,
// compare to target. Soft hit instead of hard step so gradients reach params
// even for pixels near the silhouette.

[Differentiable]
float render_pixel(no_diff float2 uv, float params[kParams])
{
    float3 ro = float3(0.0, 0.0, -3.0);
    float3 rd = normalize(float3(uv.x, uv.y, 2.0));

    float t = march(ro, rd, params);

    // Soft hit via tanh: smooth everywhere, bounded, gradients never
    // saturate to hard zero. hit=1 when t<<2, hit=0 when t>>4.
    return 0.5 - 0.5 * tanh(t - 3.0);
}

[Differentiable]
float pixel_loss(no_diff float2 uv, float params[kParams], no_diff float target)
{
    float pred = render_pixel(uv, params);
    float e = pred - target;
    return e * e;
}

// --------------------------------------------------------------------------- //
// Python entry points
// --------------------------------------------------------------------------- //

// Train one pixel: load params into a local array, run bwd_diff, scatter
// gradients back into the atomic grad tensor. SlangPy vectorizes across the
// pixel grid.
public float train_pixel(
    Tensor<float, 1> params,
    AtomicTensor<float, 1> params_grad,
    no_diff float2 uv,
    no_diff float target)
{
    float local[kParams];
    [ForceUnroll]
    for (int i = 0; i < kParams; ++i)
        local[i] = params.load(i);

    var d_local = diffPair(local);
    bwd_diff(pixel_loss)(uv, d_local, target, 1.0);

    [ForceUnroll]
    for (int i = 0; i < kParams; ++i)
        params_grad.add(i, d_local.d[i]);

    // Return the primal loss for logging.
    return pixel_loss(uv, local, target);
}

// Forward-only eval for rendering the current MLP state without tape.
public float eval_pixel(Tensor<float, 1> params, no_diff float2 uv)
{
    float local[kParams];
    [ForceUnroll]
    for (int i = 0; i < kParams; ++i)
        local[i] = params.load(i);
    return render_pixel(uv, local);
}
