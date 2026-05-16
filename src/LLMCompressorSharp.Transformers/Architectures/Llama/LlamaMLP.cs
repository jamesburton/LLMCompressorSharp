// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace LLMCompressorSharp.Transformers.Architectures.Llama;

/// <summary>
/// LLaMA gated MLP: <c>down_proj(silu(gate_proj(x)) * up_proj(x))</c>.
/// </summary>
public sealed class LlamaMLP : Module<Tensor, Tensor>
{
    private readonly Linear gate_proj;
    private readonly Linear up_proj;
    private readonly Linear down_proj;

    /// <summary>Initializes a new instance of the <see cref="LlamaMLP"/> class.</summary>
    /// <param name="hiddenSize">Outer (residual stream) dimension.</param>
    /// <param name="intermediateSize">Inner MLP dimension.</param>
    public LlamaMLP(int hiddenSize, int intermediateSize)
        : base(nameof(LlamaMLP))
    {
        if (hiddenSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");
        }

        if (intermediateSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(intermediateSize),
                intermediateSize,
                "intermediateSize must be positive.");
        }

        this.gate_proj = Linear(hiddenSize, intermediateSize, hasBias: false);
        this.up_proj = Linear(hiddenSize, intermediateSize, hasBias: false);
        this.down_proj = Linear(intermediateSize, hiddenSize, hasBias: false);
        RegisterComponents();
    }

    /// <inheritdoc />
    public override Tensor forward(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        using var gate = this.gate_proj.forward(x);
        using var up = this.up_proj.forward(x);
        using var activated = silu(gate);
        using var gated = activated.mul(up);
        return this.down_proj.forward(gated);
    }
}
