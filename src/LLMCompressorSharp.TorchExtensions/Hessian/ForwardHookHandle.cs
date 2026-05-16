// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Disposable wrapper around a TorchSharp forward-hook registration. Disposing removes the hook.
/// </summary>
/// <remarks>
/// TorchSharp 0.107.0's <c>register_forward_hook</c> returns a <c>HookRemover</c> that exposes
/// a <c>remove()</c> method. This class normalises the surface so callers always have an
/// <see cref="IDisposable"/>. The <c>removeAction</c> is captured at registration time and invoked
/// on first <see cref="Dispose"/>; subsequent calls are no-ops.
/// </remarks>
public sealed class ForwardHookHandle : IDisposable
{
    private Action? removeAction;

    /// <summary>Initializes a new instance of the <see cref="ForwardHookHandle"/> class.</summary>
    /// <param name="removeAction">Action that removes the underlying hook. Called once on <see cref="Dispose"/>.</param>
    public ForwardHookHandle(Action removeAction)
    {
        ArgumentNullException.ThrowIfNull(removeAction);
        this.removeAction = removeAction;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        var action = this.removeAction;
        this.removeAction = null;
        action?.Invoke();
    }
}
