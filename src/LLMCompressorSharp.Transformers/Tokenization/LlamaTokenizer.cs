// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using Microsoft.ML.Tokenizers;

namespace LLMCompressorSharp.Transformers.Tokenization;

/// <summary>
/// Wraps a HuggingFace BPE tokenizer from a snapshot directory into a LLaMA-family tokenizer.
/// Supports GPT-2-style vocabulary (vocab.json + merges.txt) as used by SmolLM2 and similar models.
/// </summary>
public sealed class LlamaTokenizer : IDisposable
{
    private const string VocabJsonName = "vocab.json";
    private const string MergesTxtName = "merges.txt";

    private readonly BpeTokenizer tokenizer;

    /// <summary>Initializes a new instance of the <see cref="LlamaTokenizer"/> class.</summary>
    /// <param name="snapshotDir">Directory containing <c>vocab.json</c> and <c>merges.txt</c>.</param>
    /// <exception cref="ArgumentException">If <paramref name="snapshotDir"/> is null or whitespace.</exception>
    /// <exception cref="TokenizerLoadException">If the directory or tokenizer files are missing or malformed.</exception>
    public LlamaTokenizer(string snapshotDir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(snapshotDir);

        if (!Directory.Exists(snapshotDir))
        {
            throw new TokenizerLoadException($"Snapshot directory not found: '{snapshotDir}'.");
        }

        var vocabPath = Path.Combine(snapshotDir, VocabJsonName);
        if (!File.Exists(vocabPath))
        {
            throw new TokenizerLoadException(
                $"vocab.json not found in snapshot '{snapshotDir}'. "
                + "Download the model with `huggingface-cli download <repo>` first.");
        }

        var mergesPath = Path.Combine(snapshotDir, MergesTxtName);
        if (!File.Exists(mergesPath))
        {
            throw new TokenizerLoadException(
                $"merges.txt not found in snapshot '{snapshotDir}'. "
                + "Download the model with `huggingface-cli download <repo>` first.");
        }

        try
        {
            using var vocabStream = File.OpenRead(vocabPath);
            using var mergesStream = File.OpenRead(mergesPath);
            this.tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
        }
        catch (Exception ex) when (ex is not TokenizerLoadException)
        {
            throw new TokenizerLoadException($"Failed to load tokenizer from '{snapshotDir}'.", ex);
        }
    }

    /// <summary>Gets the underlying <see cref="BpeTokenizer"/> for advanced use.</summary>
    public BpeTokenizer UnderlyingTokenizer => this.tokenizer;

    /// <summary>Encodes <paramref name="text"/> to a list of token ids.</summary>
    /// <param name="text">The input text.</param>
    /// <returns>The token ids.</returns>
    /// <exception cref="ArgumentNullException">If <paramref name="text"/> is null.</exception>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return this.tokenizer.EncodeToIds(text, considerPreTokenization: true, considerNormalization: true).ToArray();
    }

    /// <summary>Decodes a sequence of token ids back to text.</summary>
    /// <param name="ids">The token ids.</param>
    /// <returns>The decoded text, or an empty string if decoding produces no output.</returns>
    /// <exception cref="ArgumentNullException">If <paramref name="ids"/> is null.</exception>
    public string Decode(IEnumerable<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        return this.tokenizer.Decode(ids) ?? string.Empty;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // BpeTokenizer does not implement IDisposable; placeholder for future cleanup.
    }
}
