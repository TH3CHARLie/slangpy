// SPDX-License-Identifier: Apache-2.0

implementing NeuralNetworks;

__include basetypes.IModel;
__include basetypes.DiffCoopVec;

import slangpy;

/*
    Represents a linear neural network layer, i.e. a matrix-vector multiply/add A * x + b

    Takes an input with NumInputs elements, and returns NumOutputs elements.

    LinearLayer implements the multiply in software for plain arrays.
    (The CoopVec hardware-accelerated variant has been removed for the autotune
    benchmark fork — only the float/array path is exercised by the sweep.)
*/
public struct LinearLayer<
    T : IReal,
    int NumInputs,
    int NumOutputs
> : IModel<T[NumInputs], T[NumOutputs]> where T.Differential : IAtomicAddable
{
    public DiffTensor<T, 2> weights;
    public DiffTensor<T, 1> biases;

    [BackwardDerivative(backward)]
    public OutputType forward(InputType x)
    {
        OutputType y;
        for (int row = 0; row < NumOutputs; ++row)
        {
            var sum = biases.load(row);
            [ForceUnroll]
            for (int col = 0; col < NumInputs; ++col)
                sum += weights.load(row, col) * x[col];

            y[row] = sum;
        }

        return y;
    }

    public void backward(inout DifferentialPair<InputType> x, OutputType.Differential grad)
    {
        var dx = InputType.dzero();
        for (int row = 0; row < NumOutputs; ++row)
        {
            biases._grad_out.add(row, grad[row]);

            [ForceUnroll]
            for (int col = 0; col < NumInputs; ++col)
            {
                weights._grad_out.add(row, col, T.dmul(x.p[col], grad[row]));
                dx[col] = T.dadd(dx[col], T.dmul(weights.load(row, col), grad[row]));
            }
        }
        x = diffPair(x.p, dx);
    }
}
