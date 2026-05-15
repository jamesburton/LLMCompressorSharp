// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Memory;

/// <summary>
/// Ergonomic helpers around <c>torch.NewDisposeScope</c> for short-lived tensor computations.
/// </summary>
/// <remarks>
/// Encourages a discipline where every intermediate tensor in a calibration loop is freed
/// deterministically, avoiding GPU-VRAM leaks that the .NET GC cannot observe.
/// </remarks>
public static class DisposeScopeExtensions
{
    /// <summary>
    /// Runs <paramref name="body"/> inside a <c>torch.NewDisposeScope</c> and returns a scalar result.
    /// All tensors allocated by <paramref name="body"/> are disposed when the scope exits.
    /// </summary>
    /// <typeparam name="T">Result type; must be a primitive that doesn't reference a tensor.</typeparam>
    /// <param name="body">The computation. Allocate intermediates freely.</param>
    /// <returns>The scalar result.</returns>
    public static T ComputeScoped<T>(Func<T> body)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(body);
        using var scope = torch.NewDisposeScope();
        return body();
    }

    /// <summary>
    /// Runs <paramref name="body"/> inside a <c>torch.NewDisposeScope</c>, moving the returned tensor
    /// out of the scope so it survives the dispose. All other intermediates are disposed.
    /// </summary>
    /// <param name="body">The computation. The returned tensor must be the keep-alive value.</param>
    /// <returns>The result tensor, lifted out of the scope.</returns>
    public static Tensor ComputeScopedTensor(Func<Tensor> body)
    {
        ArgumentNullException.ThrowIfNull(body);
        using var scope = torch.NewDisposeScope();
        var result = body();
        return result.MoveToOuterDisposeScope();
    }
}
