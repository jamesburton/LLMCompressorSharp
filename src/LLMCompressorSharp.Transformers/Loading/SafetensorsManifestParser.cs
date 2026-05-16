// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Text.Json;

namespace LLMCompressorSharp.Transformers.Loading;

/// <summary>
/// Parses HuggingFace <c>model.safetensors.index.json</c> manifests for sharded checkpoints.
/// </summary>
public static class SafetensorsManifestParser
{
    /// <summary>Parses a JSON string into a <see cref="SafetensorsManifest"/>.</summary>
    /// <param name="json">The manifest contents.</param>
    /// <returns>The parsed manifest.</returns>
    /// <exception cref="HuggingFaceLoadException">If the JSON is malformed or missing <c>weight_map</c>.</exception>
    public static SafetensorsManifest Parse(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            throw new HuggingFaceLoadException("safetensors manifest is not valid JSON.", ex);
        }

        using (doc)
        {
            var root = doc.RootElement;
            if (!root.TryGetProperty("weight_map", out var weightMapElement)
                || weightMapElement.ValueKind != JsonValueKind.Object)
            {
                throw new HuggingFaceLoadException("safetensors manifest is missing 'weight_map'.");
            }

            var weightMap = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var entry in weightMapElement.EnumerateObject())
            {
                if (entry.Value.ValueKind != JsonValueKind.String)
                {
                    throw new HuggingFaceLoadException(
                        $"weight_map entry for '{entry.Name}' must be a string filename.");
                }

                weightMap[entry.Name] = entry.Value.GetString()!;
            }

            long? totalSize = null;
            if (root.TryGetProperty("metadata", out var meta)
                && meta.ValueKind == JsonValueKind.Object
                && meta.TryGetProperty("total_size", out var sizeProp)
                && sizeProp.ValueKind == JsonValueKind.Number)
            {
                totalSize = sizeProp.GetInt64();
            }

            return new SafetensorsManifest(weightMap, totalSize);
        }
    }
}
