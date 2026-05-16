// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Modifiers;

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Registry mapping YAML <c>type:</c> discriminators to (config subtype, factory) pairs.
/// </summary>
/// <remarks>
/// Phase 2 modifier libraries register their configs here at startup, e.g.:
/// <code>ModifierRegistry.Register&lt;GptqConfig&gt;("GPTQ", c =&gt; new GptqModifier(c));</code>
///
/// Phase 1b ships an empty registry plus the registration API and lookup methods. Tests use
/// their own registrations via the public <see cref="Register"/> method.
/// </remarks>
public static class ModifierRegistry
{
    private static readonly Dictionary<string, Registration> Registry = new(StringComparer.Ordinal);

    /// <summary>Registers a config + factory pair under a YAML type discriminator.</summary>
    /// <typeparam name="TConfig">The concrete <see cref="ModifierConfig"/> subtype.</typeparam>
    /// <param name="typeName">The YAML <c>type:</c> string (case-sensitive).</param>
    /// <param name="factory">Builds an <see cref="IModifier"/> from a config instance.</param>
    public static void Register<TConfig>(string typeName, Func<TConfig, IModifier> factory)
        where TConfig : ModifierConfig
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(typeName);
        ArgumentNullException.ThrowIfNull(factory);
        Registry[typeName] = new Registration(typeof(TConfig), c => factory((TConfig)c));
    }

    /// <summary>Removes a previously-registered type.</summary>
    /// <param name="typeName">The discriminator to unregister.</param>
    /// <returns>True if removed.</returns>
    public static bool Unregister(string typeName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(typeName);
        return Registry.Remove(typeName);
    }

    /// <summary>Clears all registrations. Intended for test isolation only.</summary>
    public static void Clear() => Registry.Clear();

    /// <summary>Resolves a registration by type discriminator.</summary>
    /// <param name="typeName">The discriminator string.</param>
    /// <returns>The registration if found, else <see langword="null"/>.</returns>
    public static Registration? Resolve(string typeName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(typeName);
        return Registry.TryGetValue(typeName, out var reg) ? reg : null;
    }

    /// <summary>A type registration.</summary>
    /// <param name="ConfigType">The concrete config CLR type.</param>
    /// <param name="Factory">A factory that turns a config into a modifier.</param>
    public sealed record Registration(Type ConfigType, Func<ModifierConfig, IModifier> Factory);
}
