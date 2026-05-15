// SPDX-License-Identifier: Apache-2.0
//
// Modular activation choices for the neural-texture benchmark.
//
// The benchmark entrypoint imports this file and combines this structural
// tunable with its own numeric shape tunables. Reflection still presents one
// combined shader-owned search space to the host tuner.

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

[Tunable]
public extern struct Act : IBenchAct = TReLU;
