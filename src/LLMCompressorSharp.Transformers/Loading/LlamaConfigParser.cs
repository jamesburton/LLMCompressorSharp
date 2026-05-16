// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Text.Json;
using LLMCompressorSharp.Transformers.Architectures.Llama;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Parses HuggingFace <c>config.json</c> into a <see cref="LlamaConfig"/>.
/// </summary>
/// <remarks>
/// Unknown fields are ignored so HF-side additions don't break loading. Required fields throw
/// <see cref="HuggingFaceLoadException"/> with the offending field name in the message.
/// </remarks>
public static class LlamaConfigParser
{
    /// <summary>Parses a JSON string into a <see cref="LlamaConfig"/>.</summary>
    /// <param name="json">The <c>config.json</c> contents.</param>
    /// <returns>The parsed config (validated).</returns>
    /// <exception cref="HuggingFaceLoadException">If the JSON is malformed or missing required fields.</exception>
    public static LlamaConfig Parse(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            throw new HuggingFaceLoadException("config.json is not valid JSON.", ex);
        }

        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                throw new HuggingFaceLoadException("config.json root must be a JSON object.");
            }

            var config = new LlamaConfig
            {
                HiddenSize = RequireInt(root, "hidden_size"),
                IntermediateSize = RequireInt(root, "intermediate_size"),
                NumHiddenLayers = RequireInt(root, "num_hidden_layers"),
                NumAttentionHeads = RequireInt(root, "num_attention_heads"),
                NumKeyValueHeads = OptionalInt(root, "num_key_value_heads") ?? RequireInt(root, "num_attention_heads"),
                VocabSize = RequireInt(root, "vocab_size"),
                MaxPositionEmbeddings = OptionalInt(root, "max_position_embeddings") ?? 2048,
                RopeTheta = OptionalFloat(root, "rope_theta") ?? 10000f,
                RmsNormEps = OptionalFloat(root, "rms_norm_eps") ?? 1e-5f,
                HiddenAct = OptionalString(root, "hidden_act") ?? "silu",
                TieWordEmbeddings = OptionalBool(root, "tie_word_embeddings") ?? true,
            };

            config.Validate();
            return config;
        }
    }

    /// <summary>Loads and parses a <c>config.json</c> file.</summary>
    /// <param name="path">Path to the config file.</param>
    /// <returns>The parsed config.</returns>
    /// <exception cref="HuggingFaceLoadException">If the file is missing, unreadable, or malformed.</exception>
    public static LlamaConfig LoadFromFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
        {
            throw new HuggingFaceLoadException($"config.json not found at '{path}'.");
        }

        string json;
        try
        {
            json = File.ReadAllText(path);
        }
        catch (IOException ex)
        {
            throw new HuggingFaceLoadException($"Failed to read config.json from '{path}'.", ex);
        }

        return Parse(json);
    }

    private static int RequireInt(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            throw new HuggingFaceLoadException($"config.json is missing required field '{field}'.");
        }

        if (prop.ValueKind != JsonValueKind.Number)
        {
            throw new HuggingFaceLoadException($"config.json field '{field}' must be a number.");
        }

        return prop.GetInt32();
    }

    private static int? OptionalInt(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return prop.ValueKind == JsonValueKind.Number ? prop.GetInt32() : null;
    }

    private static float? OptionalFloat(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return prop.ValueKind == JsonValueKind.Number ? (float)prop.GetDouble() : null;
    }

    private static string? OptionalString(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return prop.ValueKind == JsonValueKind.String ? prop.GetString() : null;
    }

    private static bool? OptionalBool(JsonElement root, string field)
    {
        if (!root.TryGetProperty(field, out var prop) || prop.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return prop.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => null,
        };
    }
}
