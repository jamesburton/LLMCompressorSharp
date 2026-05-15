// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests;

/// <summary>
/// Smoke tests verifying the toolchain is functional: TorchSharp loads its native library
/// and can allocate a CPU tensor.
/// </summary>
public class SmokeTests
{
    [Fact]
    public void TorchSharp_CanCreateCpuTensor()
    {
        using var t = zeros(2, 3);

        t.shape.Should().Equal(new long[] { 2, 3 });
        t.device_type.Should().Be(TorchSharp.DeviceType.CPU);
    }

    [Fact]
    public void TorchSharp_CanRunSimpleMatmul()
    {
        using var a = ones(2, 3);
        using var b = ones(3, 4);
        using var c = matmul(a, b);

        c.shape.Should().Equal(new long[] { 2, 4 });
        c[0, 0].item<float>().Should().Be(3.0f);
    }
}
