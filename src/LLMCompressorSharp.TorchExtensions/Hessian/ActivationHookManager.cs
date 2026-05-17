// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace LLMCompressorSharp.TorchExtensions.Hessian;

/// <summary>
/// Registers and tracks forward hooks that feed a <see cref="HessianAccumulator"/> for each layer.
/// Disposing removes every registered hook.
/// </summary>
/// <remarks>
/// Designed for the GPTQ / SparseGPT calibration loop: install hooks before iterating calibration
/// batches, run the model, dispose the manager. Each accumulator becomes the layer's Hessian.
///
/// TorchSharp 0.107.0 hook notes:
/// <list type="bullet">
/// <item><description><c>register_forward_hook</c> is defined on <c>HookableModule&lt;TPreHook, TPostHook&gt;</c>,
/// which <c>Module&lt;Tensor, Tensor&gt;</c> inherits. The post-hook delegate type is
/// <c>Func&lt;Module&lt;Tensor, Tensor&gt;, Tensor, Tensor, Tensor&gt;</c> — (module, input, output)
/// returning the (possibly modified) output.</description></item>
/// <item><description>Hooks fire via <c>.call()</c>, not <c>.forward()</c>.</description></item>
/// <item><description><c>register_forward_hook</c> returns a <c>HookRemover</c> with a <c>remove()</c>
/// method. We wrap that in a <see cref="ForwardHookHandle"/> for deterministic cleanup.</description></item>
/// </list>
/// </remarks>
public sealed class ActivationHookManager : IDisposable
{
    private readonly List<ForwardHookHandle> handles = new();
    private bool disposed;

    /// <summary>Registers a forward hook on <paramref name="layer"/> that feeds <paramref name="accumulator"/>.</summary>
    /// <param name="layer">The TorchSharp module to hook. Must be a single-input/single-output module (e.g. <c>nn.Linear</c>).</param>
    /// <param name="accumulator">The accumulator that receives each forward call's input tensor.</param>
    /// <returns>A handle that, when disposed, removes this specific hook. The manager also tracks it
    /// for batch removal via <see cref="Clear"/> or <see cref="Dispose"/>.</returns>
    /// <exception cref="ArgumentNullException">If <paramref name="layer"/> or <paramref name="accumulator"/> is null.</exception>
    public ForwardHookHandle RegisterFor(Module<Tensor, Tensor> layer, HessianAccumulator accumulator)
    {
        ArgumentNullException.ThrowIfNull(layer);
        ArgumentNullException.ThrowIfNull(accumulator);
        this.ThrowIfDisposed();

        // TorchSharp 0.107.0: TPostHook = Func<Module<Tensor, Tensor>, Tensor, Tensor, Tensor>
        // Parameters: (module, input, output) — return output unchanged to leave it unmodified.
        var hookRemover = layer.register_forward_hook((mod, input, output) =>
        {
            accumulator.Update(input);
            return output;
        });

        var handle = new ForwardHookHandle(() => hookRemover.remove());
        this.handles.Add(handle);
        return handle;
    }

    /// <summary>Removes all hooks registered through this manager and clears the internal list.</summary>
    public void Clear()
    {
        this.ThrowIfDisposed();
        foreach (var handle in this.handles)
        {
            handle.Dispose();
        }

        this.handles.Clear();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!this.disposed)
        {
            // Reuse Clear() logic but bypass ThrowIfDisposed guard since we're disposing.
            foreach (var handle in this.handles)
            {
                handle.Dispose();
            }

            this.handles.Clear();
            this.disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(this.disposed, this);
    }
}
