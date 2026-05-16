// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Compression;
using LLMCompressorSharp.Core.Modifiers;
using LLMCompressorSharp.TorchExtensions.Observers;
using LLMCompressorSharp.TorchExtensions.Quantization;
using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Algorithms.Rtn;

/// <summary>
/// Round-to-nearest weight quantization. Data-free: the weight itself is the calibration source.
/// </summary>
/// <remarks>
/// For each targeted weight:
/// <list type="number">
///   <item>Build an <see cref="Observer"/> based on <see cref="RtnConfig.Strategy"/>.</item>
///   <item>Feed the weight tensor through <c>Observer.Update</c>.</item>
///   <item>Compute scale + zero-point via <c>Observer.GetQuantParams</c>.</item>
///   <item>Fake-quantize the weight via <see cref="FakeQuantize"/> and write the result back.</item>
/// </list>
/// </remarks>
public sealed class RtnModifier : ModifierBase
{
    private readonly RtnConfig _config;

    /// <summary>Initializes a new instance of the <see cref="RtnModifier"/> class.</summary>
    /// <param name="config">The configuration.</param>
    public RtnModifier(RtnConfig config)
        : base("RTN", config?.Targets, config?.Ignore)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
    }

    /// <inheritdoc />
    protected override void OnInitialize(CompressionState state)
    {
    }

    /// <inheritdoc />
    protected override void OnEndCore(CompressionState state)
    {
        var targeted = GetTargetedNames(state).ToList();
        foreach (var name in targeted)
        {
            // Do not dispose the original tensor — the caller still owns its lifetime.
            var weight = state.NamedWeights[name];
            var observer = CreateObserver();
            observer.Update(weight);

            var quantParams = observer.GetQuantParams(_config.NumBits, _config.Symmetric);

            using var fakeQuant = ApplyFakeQuant(weight, quantParams);
            state.NamedWeights[name] = fakeQuant.detach().clone();
            quantParams.Scale.Dispose();
            quantParams.ZeroPoint.Dispose();
        }
    }

    private Observer CreateObserver()
    {
        return _config.Strategy switch
        {
            QuantizationStrategy.PerTensor => new MinMaxObserver(),
            QuantizationStrategy.PerChannel => new PerChannelMinMaxObserver(_config.ChannelAxis),
            QuantizationStrategy.PerToken => throw new NotSupportedException(
                "PerToken strategy applies to activations, not weights."),
            _ => throw new ArgumentOutOfRangeException(
                nameof(_config.Strategy),
                _config.Strategy,
                "Unsupported quantization strategy."),
        };
    }

    private Tensor ApplyFakeQuant(Tensor weight, QuantizationParameters parameters)
    {
        if (_config.Strategy == QuantizationStrategy.PerTensor)
        {
            var scale = parameters.Scale.item<float>();
            var zeroPoint = parameters.ZeroPoint.item<long>();
            return FakeQuantize.Apply(weight, scale, zeroPoint, _config.NumBits, _config.Symmetric);
        }

        return ApplyPerChannelFakeQuant(weight, parameters);
    }

    private Tensor ApplyPerChannelFakeQuant(Tensor weight, QuantizationParameters parameters)
    {
        var axis = _config.ChannelAxis;
        var scales = parameters.Scale.cpu().data<float>().ToArray();
        var zeroPoints = parameters.ZeroPoint.cpu().data<long>().ToArray();
        var channelCount = (int)weight.shape[axis];

        if (scales.Length != channelCount)
        {
            throw new InvalidOperationException(
                $"Per-channel observer returned {scales.Length} scales for {channelCount} channels.");
        }

        var resultParts = new List<Tensor>();
        for (var c = 0; c < channelCount; c++)
        {
            using var slice = weight.select(axis, c).contiguous();
            using var quantized = FakeQuantize.Apply(slice, scales[c], zeroPoints[c], _config.NumBits, _config.Symmetric);
            resultParts.Add(quantized.unsqueeze(axis).detach().clone());
        }

        try
        {
            return torch.cat(resultParts.ToArray(), axis);
        }
        finally
        {
            foreach (var p in resultParts)
            {
                p.Dispose();
            }
        }
    }
}
