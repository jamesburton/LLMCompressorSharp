// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Recipes;

/// <summary>
/// Polymorphic base for serialized modifier configurations.
/// </summary>
/// <remarks>
/// Subclasses are registered with <see cref="ModifierRegistry"/> under a short type name
/// used as the YAML <c>type:</c> discriminator.
/// </remarks>
public abstract class ModifierConfig
{
    /// <summary>Gets the short type identifier (matches the YAML <c>type:</c> field).</summary>
    public abstract string Type { get; }

    /// <summary>Gets or sets the target name patterns. Null is treated as "match all".</summary>
    public IReadOnlyList<string>? Targets { get; set; }

    /// <summary>Gets or sets the ignore name patterns. Null is treated as "ignore none".</summary>
    public IReadOnlyList<string>? Ignore { get; set; }
}
