// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.Core.Algorithms.Configs;

/// <summary>
/// A SmoothQuant mapping: the smooth-layer weight (1-D, LayerNorm-gain style) whose output
/// channels are scaled down by <c>s</c>, the balance-layer weight (2-D, Linear-style) whose
/// input channels are scaled up by <c>s</c>, and the activation tensor key that feeds the
/// balance layer.
/// </summary>
/// <param name="SmoothWeight">Name of the smooth-layer weight in <c>state.NamedWeights</c>. Must be 1-D.</param>
/// <param name="BalanceWeight">Name of the balance-layer weight in <c>state.NamedWeights</c>. Must be 2-D <c>[out, hidden]</c>.</param>
/// <param name="ActivationKey">Key in <c>state.LayerActivations</c> for the activation feeding <see cref="BalanceWeight"/>.</param>
public sealed record SmoothQuantMapping(string SmoothWeight, string BalanceWeight, string ActivationKey);
