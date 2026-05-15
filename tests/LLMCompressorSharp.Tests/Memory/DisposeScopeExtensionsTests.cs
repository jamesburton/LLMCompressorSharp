// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Memory;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.Memory;

/// <summary>
/// Tests for <see cref="DisposeScopeExtensions"/> — ergonomic helpers for short-lived tensor computations.
/// </summary>
public class DisposeScopeExtensionsTests
{
    [Fact]
    public void ComputeScoped_ReturnsResultAndDisposesIntermediates()
    {
        var result = DisposeScopeExtensions.ComputeScoped(() =>
        {
            using var a = ones(100, 100);
            using var b = ones(100, 100);
            using var c = matmul(a, b);
            return c.sum().item<float>();
        });

        // ones(100,100) @ ones(100,100) → each cell is 100; total sum = 100 * 100 * 100 = 1,000,000.
        result.Should().Be(1_000_000f);
    }

    [Fact]
    public void ComputeScopedTensor_KeepsReturnedTensorAliveOutsideScope()
    {
        // The returned tensor must survive scope teardown.
        Tensor outside = DisposeScopeExtensions.ComputeScopedTensor(() =>
        {
            var t = ones(3, 4);
            return t;
        });

        using (outside)
        {
            outside.IsInvalid.Should().BeFalse();
            outside.shape.Should().Equal(new long[] { 3, 4 });
        }
    }
}
