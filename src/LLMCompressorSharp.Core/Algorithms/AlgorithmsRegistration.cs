// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using LLMCompressorSharp.Core.Algorithms.Configs;
using LLMCompressorSharp.Core.Algorithms.Pruning;
using LLMCompressorSharp.Core.Algorithms.Rtn;
using LLMCompressorSharp.Core.Algorithms.SmoothQuant;
using LLMCompressorSharp.Core.Recipes;

namespace LLMCompressorSharp.Core.Algorithms;

/// <summary>
/// Registers all built-in algorithm config + modifier pairs with <see cref="ModifierRegistry"/>.
/// </summary>
/// <remarks>
/// The CLI invokes <see cref="RegisterAll"/> once at startup. Tests register algorithms
/// individually as needed using the singular <see cref="RegisterRtn"/> / <see cref="RegisterMagnitudePruning"/> helpers.
/// </remarks>
public static class AlgorithmsRegistration
{
    /// <summary>Registers every built-in algorithm with <see cref="ModifierRegistry"/>.</summary>
    public static void RegisterAll()
    {
        RegisterRtn();
        RegisterMagnitudePruning();
        RegisterWanda();
        RegisterSmoothQuant();
    }

    /// <summary>Registers the RTN algorithm.</summary>
    public static void RegisterRtn()
    {
        ModifierRegistry.Register<RtnConfig>("RTN", c => new RtnModifier(c));
    }

    /// <summary>Registers the magnitude pruning algorithm.</summary>
    public static void RegisterMagnitudePruning()
    {
        ModifierRegistry.Register<MagnitudePruningConfig>("MagnitudePruning", c => new MagnitudePruningModifier(c));
    }

    /// <summary>Registers the WANDA algorithm.</summary>
    public static void RegisterWanda()
    {
        ModifierRegistry.Register<WandaConfig>("WANDA", c => new WandaModifier(c));
    }

    /// <summary>Registers the SmoothQuant algorithm.</summary>
    public static void RegisterSmoothQuant()
    {
        ModifierRegistry.Register<SmoothQuantConfig>("SmoothQuant", c => new SmoothQuantModifier(c));
    }
}
