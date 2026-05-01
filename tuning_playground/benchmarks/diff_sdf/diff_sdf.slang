// SPDX-License-Identifier: Apache-2.0
//
// Differentiable SDF raymarching — fit a neural SDF (tiny MLP) to a target
// analytic sphere by backpropagating per-pixel L2 loss through a sphere-traced
// raymarch loop.
//
// The paper novelty lives here: the Slang autodiff *checkpointing schedule*
// (`[PreferRecompute]` vs `[PreferCheckpoint]`) is exposed at three independent
// call sites via `[Tunable] extern struct`. Each axis has two impls; the three
// axes compose into 2^3 = 8 link-time configurations. No macro system can
// express this cleanly.
//
// Axes:
//   SchedMLP    — schedule for mlp forward (called once per raymarch step)
//   SchedSDF    — schedule for neural SDF eval wrapper
//   SchedMarch  — schedule for the entire raymarch loop
//
// Compile-time shape (Python-driven via #define):
//   N_STEPS  — raymarch iterations (affects state cost of checkpointing)
//   W        — MLP hidden width (affects cost of recomputing)

import slangpy;

#ifndef N_STEPS
#define N_STEPS 32
#endif
#ifndef W
#define W 16
#endif

static const int kSteps = N_STEPS;
static const int kWidth = W;
static const int kInput = 3;

// Parameter layout (flat):
//   [0 .. kInput*kWidth)                     — W0 (kWidth rows × kInput cols)
//   [kInput*kWidth .. kInput*kWidth+kWidth)  — b0 (kWidth)
//   [... .. ... + kWidth)                    — W1 (1 row × kWidth cols)
//   [... .. ... + 1)                         — b1 (1)
static const int kParams = kInput * kWidth + kWidth + kWidth + 1;

// --------------------------------------------------------------------------- //
// MLP forward — schedule axis #1
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

public interface IMLPSched
{
    [Differentiable]
    static float forward(no_diff float3 p, float params[kParams]);
}

public struct MLPRecompute : IMLPSched
{
    [Differentiable]
    [PreferRecompute]
    public static float forward(no_diff float3 p, float params[kParams])
    {
        return mlp_body(p, params);
    }
}

public struct MLPCheckpoint : IMLPSched
{
    [Differentiable]
    [PreferCheckpoint]
    public static float forward(no_diff float3 p, float params[kParams])
    {
        return mlp_body(p, params);
    }
}

[Tunable] public extern struct SchedMLP : IMLPSched = MLPCheckpoint;

// --------------------------------------------------------------------------- //
// Neural SDF eval wrapper — schedule axis #2
// --------------------------------------------------------------------------- //
//
// In principle this is just a pass-through around SchedMLP. Wrapping it lets
// us schedule the SDF eval independently of the MLP — useful when the MLP is
// cheap but the enclosing per-step state (ray position, normal estimate, etc.)
// is expensive to keep live.

[Differentiable]
float sdf_body(no_diff float3 p, float params[kParams])
{
    return SchedMLP::forward(p, params);
}

public interface ISDFSched
{
    [Differentiable]
    static float eval(no_diff float3 p, float params[kParams]);
}

public struct SDFRecompute : ISDFSched
{
    [Differentiable]
    [PreferRecompute]
    public static float eval(no_diff float3 p, float params[kParams])
    {
        return sdf_body(p, params);
    }
}

public struct SDFCheckpoint : ISDFSched
{
    [Differentiable]
    [PreferCheckpoint]
    public static float eval(no_diff float3 p, float params[kParams])
    {
        return sdf_body(p, params);
    }
}

[Tunable] public extern struct SchedSDF : ISDFSched = SDFCheckpoint;

// --------------------------------------------------------------------------- //
// Raymarch loop — schedule axis #3
// --------------------------------------------------------------------------- //
//
// Fixed-iteration sphere-tracing loop. No early-out so the loop is uniform;
// this simplifies the autodiff state layout (important for checkpointing).
// Final `t` is the marched distance.

[Differentiable]
float march_body(no_diff float3 ro, no_diff float3 rd, float params[kParams])
{
    float t = 0.0;
    [MaxIters(kSteps)]
    for (int i = 0; i < kSteps; ++i)
    {
        float3 p = ro + rd * t;
        float d = SchedSDF::eval(p, params);
        // Smooth-saturating step: tanh keeps |d| bounded without a hard
        // cutoff that zeros gradients at the boundary.
        t += 3.0 * tanh(d / 3.0);
    }
    return t;
}

public interface IMarchSched
{
    [Differentiable]
    static float march(no_diff float3 ro, no_diff float3 rd, float params[kParams]);
}

public struct MarchRecompute : IMarchSched
{
    [Differentiable]
    [PreferRecompute]
    public static float march(no_diff float3 ro, no_diff float3 rd, float params[kParams])
    {
        return march_body(ro, rd, params);
    }
}

public struct MarchCheckpoint : IMarchSched
{
    [Differentiable]
    [PreferCheckpoint]
    public static float march(no_diff float3 ro, no_diff float3 rd, float params[kParams])
    {
        return march_body(ro, rd, params);
    }
}

[Tunable] public extern struct SchedMarch : IMarchSched = MarchCheckpoint;

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

    float t = SchedMarch::march(ro, rd, params);

    // Soft hit via tanh — smooth everywhere, bounded, gradients never
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
// gradients back into the atomic grad tensor. SlangPy vectorises across the
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

// Forward-only eval — for rendering the current MLP state without tape.
public float eval_pixel(Tensor<float, 1> params, no_diff float2 uv)
{
    float local[kParams];
    [ForceUnroll]
    for (int i = 0; i < kParams; ++i)
        local[i] = params.load(i);
    return render_pixel(uv, local);
}
