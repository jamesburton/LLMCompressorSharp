// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.Core.Compression;

/// <summary>
/// Mutable state shared with every IModifier across a compression session.
/// </summary>
/// <remarks>
/// Phase 1b exposes the named-weights dictionary and per-batch sample. Phase 3 will add a
/// helper to extract NamedWeights from a TorchSharp Module; until then, callers populate it
/// directly (typically from a safetensors load).
/// </remarks>
public sealed class CompressionState
{
    private readonly Dictionary<string, Tensor> _namedWeights;

    /// <summary>Initializes a new instance of the <see cref="CompressionState"/> class.</summary>
    /// <param name="namedWeights">The initial named-weights dictionary. The session takes ownership.</param>
    /// <param name="rngSeed">Deterministic seed for any RNG-using modifier.</param>
    public CompressionState(IReadOnlyDictionary<string, Tensor> namedWeights, long rngSeed = 0L)
    {
        ArgumentNullException.ThrowIfNull(namedWeights);
        _namedWeights = new Dictionary<string, Tensor>(namedWeights);
        RngSeed = rngSeed;
    }

    /// <summary>Gets the named-weights dictionary. Modifiers may replace values in place.</summary>
    public IDictionary<string, Tensor> NamedWeights => _namedWeights;

    /// <summary>Gets or sets the current calibration batch tensor, or <see langword="null"/> outside a batch.</summary>
    public Tensor? CurrentBatch { get; set; }

    /// <summary>Gets or sets the zero-based index of the current calibration batch.</summary>
    public int CurrentBatchIndex { get; set; }

    /// <summary>
    /// Gets or sets per-layer activation tensors collected during calibration, or
    /// <see langword="null"/> if no activations are routed to the session.
    /// </summary>
    /// <remarks>
    /// Phase 2b modifiers (WANDA, SmoothQuant) read activations from this dictionary using
    /// the weight name stripped of its <c>.weight</c> suffix as the key. Phase 3 will populate
    /// this dictionary automatically via forward hooks on a TorchSharp <c>Module</c>; until
    /// then, tests populate it directly.
    /// </remarks>
    public IDictionary<string, Tensor>? LayerActivations { get; set; }

    /// <summary>
    /// Gets or sets the named module dictionary used by hook-based modifiers (e.g. GPTQ, SparseGPT).
    /// Keys match those in <see cref="NamedWeights"/>, with the <c>.weight</c> suffix stripped.
    /// When null, hook-based modifiers are not supported.
    /// </summary>
    /// <remarks>
    /// Phase 4b populates this from the LLaMA module hierarchy via
    /// <c>model.named_modules()</c> filtered to <c>Linear</c> layers.
    /// </remarks>
    public IDictionary<string, Module<Tensor, Tensor>>? NamedModules { get; set; }

    /// <summary>Gets the deterministic RNG seed assigned to this session.</summary>
    public long RngSeed { get; }
}
