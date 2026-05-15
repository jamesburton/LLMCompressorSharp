// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

namespace LLMCompressorSharp.TorchExtensions.Memory;

/// <summary>
/// Optional NVML-backed GPU memory statistics. The full P/Invoke implementation is deferred;
/// this stub returns <see langword="null"/> for all accessors so callers can use it conditionally.
/// </summary>
/// <remarks>
/// Tracked upstream as PR_TO_TORCHSHARP P-001 (would expose <c>torch.cuda.memory_allocated</c> etc.).
/// </remarks>
// TODO(PR_TO_TORCHSHARP P-001): Replace with torch.cuda.memory_allocated/reserved/stats when upstream lands.
public static class NvmlMemoryStats
{
    /// <summary>Gets a value indicating whether NVML is available in this process.</summary>
    public static bool IsAvailable => false;

    /// <summary>
    /// Returns the number of bytes currently allocated on the given CUDA device, or
    /// <see langword="null"/> if NVML is unavailable (e.g. CPU-only build, NVML not installed).
    /// </summary>
    /// <param name="deviceIndex">CUDA device index.</param>
    /// <returns>Allocated bytes, or <see langword="null"/>.</returns>
    public static long? GetAllocatedBytes(int deviceIndex = 0)
    {
        _ = deviceIndex;
        return null;
    }

    /// <summary>
    /// Returns the number of bytes reserved by the CUDA allocator on the given device,
    /// or <see langword="null"/> if NVML is unavailable.
    /// </summary>
    /// <param name="deviceIndex">CUDA device index.</param>
    /// <returns>Reserved bytes, or <see langword="null"/>.</returns>
    public static long? GetReservedBytes(int deviceIndex = 0)
    {
        _ = deviceIndex;
        return null;
    }
}
