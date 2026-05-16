// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace LLMCompressorSharp.Transformers.Tokenization;

/// <summary>
/// Wraps a HuggingFace BPE tokenizer from a snapshot directory.
/// </summary>
/// <remarks>
/// Supports two layouts:
/// <list type="number">
///   <item>HuggingFace single-file: <c>tokenizer.json</c> with embedded BPE vocab + merges (SmolLM2, GPT-2, etc.).</item>
///   <item>Two-file: <c>vocab.json</c> + <c>merges.txt</c> (older HF mirrors, some CodeGen/Phi releases).</item>
/// </list>
/// Single-file is preferred; the two-file path is the fallback. SentencePiece (<c>tokenizer.model</c>)
/// is not supported in Phase 3c — coming in a later phase when we target Meta Llama-2/3.
/// </remarks>
public sealed class LlamaTokenizer : IDisposable
{
    private const string TokenizerJsonName = "tokenizer.json";
    private const string VocabJsonName = "vocab.json";
    private const string MergesTxtName = "merges.txt";

    private readonly BpeTokenizer tokenizer;

    /// <summary>Initializes a new instance of the <see cref="LlamaTokenizer"/> class.</summary>
    /// <param name="snapshotDir">Directory containing <c>tokenizer.json</c> or <c>vocab.json</c>+<c>merges.txt</c>.</param>
    /// <exception cref="ArgumentException">If <paramref name="snapshotDir"/> is null or whitespace.</exception>
    /// <exception cref="TokenizerLoadException">If the directory or tokenizer files are missing or malformed.</exception>
    public LlamaTokenizer(string snapshotDir)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(snapshotDir);

        if (!Directory.Exists(snapshotDir))
        {
            throw new TokenizerLoadException($"Snapshot directory not found: '{snapshotDir}'.");
        }

        var tokenizerJsonPath = Path.Combine(snapshotDir, TokenizerJsonName);
        var vocabPath = Path.Combine(snapshotDir, VocabJsonName);
        var mergesPath = Path.Combine(snapshotDir, MergesTxtName);

        if (File.Exists(tokenizerJsonPath))
        {
            this.tokenizer = LoadFromTokenizerJson(tokenizerJsonPath);
        }
        else if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            this.tokenizer = LoadFromVocabAndMerges(vocabPath, mergesPath);
        }
        else
        {
            throw new TokenizerLoadException(
                $"No tokenizer found in snapshot '{snapshotDir}'. "
                + $"Expected '{TokenizerJsonName}' or '{VocabJsonName}'+'{MergesTxtName}'.");
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

    private static BpeTokenizer LoadFromTokenizerJson(string tokenizerJsonPath)
    {
        try
        {
            using var stream = File.OpenRead(tokenizerJsonPath);
            using var doc = JsonDocument.Parse(stream);

            if (!doc.RootElement.TryGetProperty("model", out var modelElement))
            {
                throw new TokenizerLoadException(
                    $"tokenizer.json at '{tokenizerJsonPath}' has no 'model' field.");
            }

            if (modelElement.TryGetProperty("type", out var typeElement)
                && typeElement.ValueKind == JsonValueKind.String
                && !string.Equals(typeElement.GetString(), "BPE", StringComparison.Ordinal))
            {
                throw new TokenizerLoadException(
                    $"tokenizer.json model type '{typeElement.GetString()}' is not supported (only 'BPE').");
            }

            if (!modelElement.TryGetProperty("vocab", out var vocabElement)
                || vocabElement.ValueKind != JsonValueKind.Object)
            {
                throw new TokenizerLoadException("tokenizer.json model has no 'vocab' object.");
            }

            if (!modelElement.TryGetProperty("merges", out var mergesElement)
                || mergesElement.ValueKind != JsonValueKind.Array)
            {
                throw new TokenizerLoadException("tokenizer.json model has no 'merges' array.");
            }

            using var vocabMemory = new MemoryStream();
            using (var writer = new Utf8JsonWriter(vocabMemory))
            {
                vocabElement.WriteTo(writer);
            }

            vocabMemory.Position = 0;

            using var mergesMemory = new MemoryStream();
            using (var sw = new StreamWriter(mergesMemory, new UTF8Encoding(false), leaveOpen: true))
            {
                sw.WriteLine("#version: 0.2");
                foreach (var merge in mergesElement.EnumerateArray())
                {
                    if (merge.ValueKind == JsonValueKind.String)
                    {
                        sw.WriteLine(merge.GetString());
                    }
                    else if (merge.ValueKind == JsonValueKind.Array)
                    {
                        // Some tokenizers store merges as [a, b] pairs instead of "a b" strings.
                        var parts = new List<string>();
                        foreach (var part in merge.EnumerateArray())
                        {
                            if (part.ValueKind == JsonValueKind.String)
                            {
                                parts.Add(part.GetString() ?? string.Empty);
                            }
                        }

                        sw.WriteLine(string.Join(' ', parts));
                    }
                }
            }

            mergesMemory.Position = 0;
            return BpeTokenizer.Create(vocabMemory, mergesMemory);
        }
        catch (Exception ex) when (ex is not TokenizerLoadException)
        {
            throw new TokenizerLoadException($"Failed to load tokenizer from '{tokenizerJsonPath}'.", ex);
        }
    }

    private static BpeTokenizer LoadFromVocabAndMerges(string vocabPath, string mergesPath)
    {
        try
        {
            using var vocabStream = File.OpenRead(vocabPath);
            using var mergesStream = File.OpenRead(mergesPath);
            return BpeTokenizer.Create(vocabStream, mergesStream);
        }
        catch (Exception ex) when (ex is not TokenizerLoadException)
        {
            throw new TokenizerLoadException(
                $"Failed to load tokenizer from '{vocabPath}' + '{mergesPath}'.", ex);
        }
    }
}
