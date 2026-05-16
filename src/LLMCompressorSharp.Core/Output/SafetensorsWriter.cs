// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Compression;
using TorchSharp;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Core.Output;

/// <summary>
/// Saves and loads <see cref="CompressionState.NamedWeights"/> as safetensors files.
/// </summary>
/// <remarks>
/// Uses <c>Safetensors.SaveStateDict</c> and <c>Safetensors.LoadStateDict</c>
/// from <c>TorchSharp.PyBridge</c> directly — no wrapper module required, and tensor key
/// names (including dots) are preserved verbatim by the safetensors format.
/// </remarks>
public static class SafetensorsWriter
{
    /// <summary>Saves the named weights of <paramref name="state"/> to <paramref name="path"/>.</summary>
    /// <param name="state">The compression state whose <see cref="CompressionState.NamedWeights"/> are saved.</param>
    /// <param name="path">Output safetensors file path. The parent directory must exist.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="state"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="path"/> is null or whitespace.</exception>
    public static void Save(CompressionState state, string path)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        // Safetensors.SaveStateDict requires Dictionary<string, Tensor>, not IDictionary.
        var dict = state.NamedWeights as Dictionary<string, Tensor>
            ?? new Dictionary<string, Tensor>(state.NamedWeights);

        Safetensors.SaveStateDict(path, dict);
    }

    /// <summary>Loads a safetensors file into a dictionary of named tensors.</summary>
    /// <param name="path">Input safetensors file path.</param>
    /// <returns>The loaded tensors, keyed by name. The caller is responsible for disposing the returned tensors.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="path"/> is null or whitespace.</exception>
    /// <exception cref="FileNotFoundException">Thrown when <paramref name="path"/> does not exist on disk.</exception>
    public static IDictionary<string, Tensor> Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
        {
            throw new FileNotFoundException("Safetensors file not found.", path);
        }

        return Safetensors.LoadStateDict(path);
    }
}
