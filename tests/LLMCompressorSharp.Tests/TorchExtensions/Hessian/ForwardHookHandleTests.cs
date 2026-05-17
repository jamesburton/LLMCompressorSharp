// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using FluentAssertions;
using LLMCompressorSharp.TorchExtensions.Hessian;
using Xunit;

namespace LLMCompressorSharp.Tests.TorchExtensions.Hessian;

/// <summary>
/// Tests for <see cref="ForwardHookHandle"/>.
/// </summary>
public class ForwardHookHandleTests
{
    [Fact]
    public void Dispose_InvokesRemoveAction()
    {
        var called = 0;
        var handle = new ForwardHookHandle(() => called++);
        handle.Dispose();
        called.Should().Be(1);
    }

    [Fact]
    public void Dispose_CalledTwice_OnlyInvokesActionOnce()
    {
        var called = 0;
        var handle = new ForwardHookHandle(() => called++);
        handle.Dispose();
        handle.Dispose();
        called.Should().Be(1);
    }

    [Fact]
    public void Constructor_NullAction_Throws()
    {
        var act = () => new ForwardHookHandle(null!);
        act.Should().Throw<ArgumentNullException>();
    }
}
