// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Text.RegularExpressions;

namespace LLMCompressorSharp.Core.Modifiers;

/// <summary>
/// Name-pattern matching for selecting which named weights a modifier targets.
/// </summary>
/// <remarks>
/// A name matches if targets is empty (match-all) or any target pattern matches, and no ignore
/// pattern matches. Patterns are glob-style — * matches any sequence of characters, including
/// path separators (dots, slashes).
/// </remarks>
public static class TargetMatcher
{
    /// <summary>Returns true if <paramref name="name"/> matches the target/ignore patterns.</summary>
    /// <param name="name">The fully qualified weight or module name (e.g. <c>model.layers.0.q_proj.weight</c>).</param>
    /// <param name="targets">Target patterns; empty means "match any".</param>
    /// <param name="ignore">Ignore patterns; any match excludes the name.</param>
    /// <returns>True if the name should be processed.</returns>
    public static bool Matches(string name, IReadOnlyList<string> targets, IReadOnlyList<string> ignore)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(targets);
        ArgumentNullException.ThrowIfNull(ignore);

        foreach (var pattern in ignore)
        {
            if (MatchesPattern(name, pattern))
            {
                return false;
            }
        }

        if (targets.Count == 0)
        {
            return true;
        }

        foreach (var pattern in targets)
        {
            if (MatchesPattern(name, pattern))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Filters <paramref name="names"/> through <see cref="Matches"/>.</summary>
    /// <param name="names">Candidate names.</param>
    /// <param name="targets">Target patterns.</param>
    /// <param name="ignore">Ignore patterns.</param>
    /// <returns>The subset that matches.</returns>
    public static IEnumerable<string> Filter(
        IEnumerable<string> names,
        IReadOnlyList<string> targets,
        IReadOnlyList<string> ignore)
    {
        ArgumentNullException.ThrowIfNull(names);
        foreach (var n in names)
        {
            if (Matches(n, targets, ignore))
            {
                yield return n;
            }
        }
    }

    private static bool MatchesPattern(string name, string pattern)
    {
        if (string.IsNullOrEmpty(pattern))
        {
            return false;
        }

        if (!pattern.Contains('*'))
        {
            return string.Equals(name, pattern, StringComparison.Ordinal);
        }

        var regex = "^" + Regex.Escape(pattern).Replace("\\*", ".*") + "$";
        return Regex.IsMatch(name, regex);
    }
}
