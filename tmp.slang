[__AttributeUsage(_AttributeTargets.Var)]
struct MaxValueAttribute
{
    int value;
    string description;
};

[MaxValue(12, "the scale factor")]
uniform int scaleFactor;

[__AttributeUsage(_AttributeTargets.Struct)]
struct TunableAttribute
{
}


interface IFoo {
    float exec(float a, int b);
}

struct FooSlow : IFoo {
    float exec(float a, int b) {
        float result = 0.0;
        for (int i = 0; i < b; i++) {
            result += a;
        }
        return result;
    }
}

struct FooFast : IFoo{
    float exec(float a, int b) {
        return sin(a) * b;
    }
}

[Tunable]
extern struct TunableFoo: IFoo = FooSlow;

float add(float a, int b)
{
    TunableFoo foo;
    return foo.exec(a, b);
}
