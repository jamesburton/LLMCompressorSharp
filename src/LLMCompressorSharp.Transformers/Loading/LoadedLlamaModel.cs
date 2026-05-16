// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Transformers.Architectures.Llama;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// The result of loading a LLaMA model from HuggingFace.
/// </summary>
/// <param name="Model">The instantiated model with weights loaded.</param>
/// <param name="Config">The parsed config.</param>
/// <param name="SnapshotPath">The on-disk snapshot directory the model was loaded from.</param>
public sealed record LoadedLlamaModel(
    LlamaForCausalLM Model,
    LlamaConfig Config,
    string SnapshotPath);
