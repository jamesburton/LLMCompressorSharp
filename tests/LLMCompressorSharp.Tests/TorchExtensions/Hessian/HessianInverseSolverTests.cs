// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Hessian;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Tests.TorchExtensions.Hessian;

/// <summary>
/// Tests for <see cref="HessianInverseSolver"/> — damped Cholesky inversion.
/// </summary>
public class HessianInverseSolverTests
{
    [Fact]
    public void Compute_IdentityMatrix_ReturnsIdentity()
    {
        // Hinv of identity (already perfectly conditioned) should be identity.
        using var h = eye(4, dtype: ScalarType.Float32);
        using var hinv = HessianInverseSolver.Compute(h, dampingFrac: 0.0f);

        // Hinv should be close to identity.
        using var expected = eye(4, dtype: ScalarType.Float32);
        using var diff = hinv.sub(expected).abs();
        diff.max().item<float>().Should().BeLessThan(1e-5f);
    }

    [Fact]
    public void Compute_ScaledIdentity_ReturnsInverseScaledIdentity()
    {
        // H = 2*I → Hinv = 0.5*I
        using var h = eye(3, dtype: ScalarType.Float32).mul(2f);
        using var hinv = HessianInverseSolver.Compute(h, dampingFrac: 0.0f);

        using var expected = eye(3, dtype: ScalarType.Float32).mul(0.5f);
        using var diff = hinv.sub(expected).abs();
        diff.max().item<float>().Should().BeLessThan(1e-5f);
    }

    [Fact]
    public void Compute_WithDamping_ProducesFiniteOutput()
    {
        // A rank-deficient matrix (not invertible without damping).
        // [[1, 1], [1, 1]] has rank 1. Damping regularises it.
        using var h = ones(2, 2, dtype: ScalarType.Float32);
        using var hinv = HessianInverseSolver.Compute(h, dampingFrac: 0.01f);

        // With damping, result should be finite.
        hinv.isfinite().all().item<bool>().Should().BeTrue();
    }

    [Fact]
    public void Compute_IllConditionedMatrix_FallsBackToPinv()
    {
        // A very ill-conditioned matrix that even with mild damping may fail Cholesky.
        // Use a near-zero diagonal to create a challenging case.
        using var h = eye(4, dtype: ScalarType.Float32).mul(1e-10f);
        using var hinv = HessianInverseSolver.Compute(h, dampingFrac: 0.0f);

        // Pseudoinverse fallback should produce finite output.
        hinv.isfinite().all().item<bool>().Should().BeTrue();
    }

    [Fact]
    public void Compute_NullHessian_Throws()
    {
        var act = () => HessianInverseSolver.Compute(null!, dampingFrac: 0.01f);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Compute_NonSquareHessian_Throws()
    {
        using var h = ones(3, 4, dtype: ScalarType.Float32);
        var act = () => HessianInverseSolver.Compute(h, dampingFrac: 0.01f);
        act.Should().Throw<ArgumentException>().WithMessage("*square*");
    }

    [Fact]
    public void Compute_OutputIsFloat32()
    {
        // Even if Hinv diverges in precision, it should remain float32.
        using var h = eye(3, dtype: ScalarType.Float32);
        using var hinv = HessianInverseSolver.Compute(h, dampingFrac: 0.01f);
        hinv.dtype.Should().Be(ScalarType.Float32);
    }

    [Fact]
    public void Compute_OutputDoesNotAliasInput()
    {
        // Caller must be able to dispose H without affecting Hinv.
        using var h = eye(3, dtype: ScalarType.Float32);
        var hinv = HessianInverseSolver.Compute(h, dampingFrac: 0.01f);
        h.Dispose();

        // Hinv should still be readable.
        hinv.isfinite().all().item<bool>().Should().BeTrue();
        hinv.Dispose();
    }
}
