// --- Activation interface and implementations ---

interface IActivation {
    float activate(float x);
    float derivative(float x);
}

struct ReLU : IActivation {
    float activate(float x) {
        return max(0.0, x);
    }
    float derivative(float x) {
        return x > 0.0 ? 1.0 : 0.0;
    }
}

struct LeakyReLU : IActivation {
    static const float slope = 0.01;

    float activate(float x) {
        return x > 0.0 ? x : slope * x;
    }
    float derivative(float x) {
        return x > 0.0 ? 1.0 : slope;
    }
}

struct GELU : IActivation {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float activate(float x) {
        float c = 0.7978845608; // sqrt(2/pi)
        float inner = c * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1.0 + tanh(inner));
    }
    float derivative(float x) {
        float c = 0.7978845608;
        float x3 = x * x * x;
        float inner = c * (x + 0.044715 * x3);
        float t = tanh(inner);
        float sech2 = 1.0 - t * t;
        float dinner = c * (1.0 + 3.0 * 0.044715 * x * x);
        return 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner;
    }
}

struct SiLU : IActivation {
    // x * sigmoid(x)
    float activate(float x) {
        float s = 1.0 / (1.0 + exp(-x));
        return x * s;
    }
    float derivative(float x) {
        float s = 1.0 / (1.0 + exp(-x));
        return s * (1.0 + x * (1.0 - s));
    }
}

[Tunable]
extern struct TunableActivation : IActivation = ReLU;

// --- Reduction interface and implementations ---

interface IReduce {
    float identity();
    float combine(float a, float b);
    float finalize(float result, int count);
}

struct SumReduce : IReduce {
    float identity() { return 0.0; }
    float combine(float a, float b) { return a + b; }
    float finalize(float result, int count) { return result; }
}

struct MaxReduce : IReduce {
    float identity() { return -3.402823466e+38; } // -FLT_MAX
    float combine(float a, float b) { return max(a, b); }
    float finalize(float result, int count) { return result; }
}

struct MeanReduce : IReduce {
    float identity() { return 0.0; }
    float combine(float a, float b) { return a + b; }
    float finalize(float result, int count) { return result / float(count); }
}

[Tunable]
extern struct TunableReduce : IReduce = SumReduce;

// --- Pipeline functions ---

// Apply activation element-wise: data[i] = activate(data[i])
float activate_element(float x)
{
    TunableActivation act;
    return act.activate(x);
}

// Compute activation derivative element-wise
float activate_derivative(float x)
{
    TunableActivation act;
    return act.derivative(x);
}

// Reduce an array of values using the tunable reduction
float reduce_values(float a, float b)
{
    TunableReduce red;
    return red.combine(a, b);
}

// Full pipeline: activate then reduce
// Applies activation to each element, then reduces to a single value
float transform_reduce(float v0, float v1, float v2, float v3)
{
    TunableActivation act;
    TunableReduce red;

    float r = red.identity();
    r = red.combine(r, act.activate(v0));
    r = red.combine(r, act.activate(v1));
    r = red.combine(r, act.activate(v2));
    r = red.combine(r, act.activate(v3));
    return red.finalize(r, 4);
}
