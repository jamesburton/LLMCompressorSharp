// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Globalization;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Parses YAML recipe documents into <see cref="Recipe"/> instances, dispatching modifier
/// configs through <see cref="ModifierRegistry"/> via the YAML <c>type:</c> discriminator.
/// </summary>
public static class RecipeParser
{
    /// <summary>Parses a YAML string into a <see cref="Recipe"/>.</summary>
    /// <param name="yaml">The recipe YAML text.</param>
    /// <returns>The parsed recipe.</returns>
    /// <exception cref="RecipeParseException">If the YAML is malformed or references an unregistered modifier type.</exception>
    public static Recipe Parse(string yaml)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(yaml);

        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .WithTypeConverter(new ModifierConfigConverter())
            .Build();

        try
        {
            return deserializer.Deserialize<Recipe>(yaml) ?? new Recipe();
        }
        catch (RecipeParseException)
        {
            // Propagate converter exceptions directly without re-wrapping.
            throw;
        }
        catch (YamlException ex) when (FindInner<RecipeParseException>(ex) is { } inner)
        {
            // YamlDotNet 16.x wraps converter exceptions inside YamlException; unwrap them.
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(inner).Throw();
            throw; // unreachable, satisfies compiler
        }
        catch (YamlException ex)
        {
            throw new RecipeParseException("Failed to parse recipe YAML.", ex);
        }
    }

    /// <summary>Walks the inner-exception chain to find an exception of type <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">The exception type to locate.</typeparam>
    /// <param name="ex">The root exception to search from.</param>
    /// <returns>The first matching inner exception, or <see langword="null"/>.</returns>
    private static T? FindInner<T>(Exception? ex)
        where T : Exception
    {
        while (ex is not null)
        {
            if (ex is T found)
            {
                return found;
            }

            ex = ex.InnerException;
        }

        return null;
    }

    private sealed class ModifierConfigConverter : IYamlTypeConverter
    {
        public bool Accepts(Type type) => type == typeof(ModifierConfig);

        // YamlDotNet 16.x signature: ReadYaml(IParser, Type, ObjectDeserializer)
        public object? ReadYaml(IParser parser, Type type, ObjectDeserializer rootDeserializer)
        {
            parser.Consume<MappingStart>();
            var fields = new Dictionary<string, object?>();
            string? typeName = null;
            while (!parser.TryConsume<MappingEnd>(out _))
            {
                var key = parser.Consume<Scalar>().Value;
                if (parser.Current is Scalar s)
                {
                    parser.MoveNext();
                    if (key == "type")
                    {
                        typeName = s.Value;
                    }
                    else
                    {
                        fields[key] = s.Value;
                    }
                }
                else if (parser.Current is SequenceStart)
                {
                    parser.Consume<SequenceStart>();
                    var items = new List<string>();
                    while (!parser.TryConsume<SequenceEnd>(out _))
                    {
                        items.Add(parser.Consume<Scalar>().Value);
                    }

                    fields[key] = items;
                }
                else
                {
                    parser.SkipThisAndNestedEvents();
                }
            }

            if (string.IsNullOrEmpty(typeName))
            {
                throw new RecipeParseException("Each modifier must specify a `type:` field.");
            }

            var registration = ModifierRegistry.Resolve(typeName)
                ?? throw new RecipeParseException(
                    $"Modifier type '{typeName}' is not registered. Did you call ModifierRegistry.Register<...>?");

            var config = (ModifierConfig)Activator.CreateInstance(registration.ConfigType)!;
            PopulateConfig(config, fields);
            return config;
        }

        // YamlDotNet 16.x signature: WriteYaml(IEmitter, object?, Type, ObjectSerializer)
        public void WriteYaml(IEmitter emitter, object? value, Type type, ObjectSerializer serializer)
        {
            throw new NotSupportedException("Recipe serialization is not implemented in Phase 1b.");
        }

        private static void PopulateConfig(ModifierConfig config, Dictionary<string, object?> fields)
        {
            foreach (var (key, raw) in fields)
            {
                var prop = config.GetType().GetProperty(ToPascalCase(key));
                if (prop is null || !prop.CanWrite)
                {
                    continue;
                }

                if (raw is List<string> list)
                {
                    prop.SetValue(config, list);
                }
                else if (raw is string s)
                {
                    var converted = ConvertScalar(s, prop.PropertyType);
                    prop.SetValue(config, converted);
                }
            }
        }

        private static string ToPascalCase(string snake)
        {
            if (string.IsNullOrEmpty(snake))
            {
                return snake;
            }

            var sb = new System.Text.StringBuilder(snake.Length);
            var nextUpper = true;
            foreach (var c in snake)
            {
                if (c == '_')
                {
                    nextUpper = true;
                    continue;
                }

                sb.Append(nextUpper ? char.ToUpper(c, CultureInfo.InvariantCulture) : c);
                nextUpper = false;
            }

            return sb.ToString();
        }

        private static object? ConvertScalar(string s, Type targetType)
        {
            if (targetType == typeof(string))
            {
                return s;
            }

            if (targetType == typeof(int) || targetType == typeof(int?))
            {
                return int.Parse(s, CultureInfo.InvariantCulture);
            }

            if (targetType == typeof(long) || targetType == typeof(long?))
            {
                return long.Parse(s, CultureInfo.InvariantCulture);
            }

            if (targetType == typeof(float) || targetType == typeof(float?))
            {
                return float.Parse(s, CultureInfo.InvariantCulture);
            }

            if (targetType == typeof(double) || targetType == typeof(double?))
            {
                return double.Parse(s, CultureInfo.InvariantCulture);
            }

            if (targetType == typeof(bool) || targetType == typeof(bool?))
            {
                return bool.Parse(s);
            }

            return s;
        }
    }
}
