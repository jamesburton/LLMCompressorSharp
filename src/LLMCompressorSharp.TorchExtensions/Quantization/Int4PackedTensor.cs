// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using TorchSharp;
using static TorchSharp.torch;

namespace LLMCompressorSharp.TorchExtensions.Quantization;

/// <summary>
/// FP32-backed simulation of a packed INT4 tensor. The in-memory representation is FP32 for
/// arithmetic convenience; the <see cref="PackedBytes"/> view returns one byte per pair of INT4
/// values to make storage layout match real INT4 backends.
/// </summary>
/// <remarks>
/// True packed INT4 in TorchSharp is tracked as PR_TO_TORCHSHARP R-003 and likely requires a fork.
/// </remarks>
// TODO(PR_TO_TORCHSHARP R-003): Replace with native QInt4 dtype when upstream / fork lands.
public sealed class Int4PackedTensor
{
    private readonly sbyte[] _ints;
    private readonly float _scale;
    private readonly long _zeroPoint;
    private readonly bool _symmetric;

    private Int4PackedTensor(sbyte[] ints, float scale, long zeroPoint, bool symmetric)
    {
        _ints = ints;
        _scale = scale;
        _zeroPoint = zeroPoint;
        _symmetric = symmetric;
    }

    /// <summary>Gets the integer-grid quantization step.</summary>
    public float Scale => _scale;

    /// <summary>Gets the zero-point offset for asymmetric quantization.</summary>
    public long ZeroPoint => _zeroPoint;

    /// <summary>Gets a value indicating whether the original quantization was symmetric.</summary>
    public bool Symmetric => _symmetric;

    /// <summary>Gets the number of int4 elements.</summary>
    public int ElementCount => _ints.Length;

    /// <summary>
    /// Gets the packed byte representation: two int4 values per byte (high nibble = even index, low nibble = odd index).
    /// </summary>
    public byte[] PackedBytes
    {
        get
        {
            var n = _ints.Length;
            var bytes = new byte[(n + 1) / 2];
            for (var i = 0; i < n; i++)
            {
                // We pack the post-zero-point integer; subtraction is reversed in Dequantize.
                var stored = (byte)(_ints[i] & 0x0F);
                if ((i & 1) == 0)
                {
                    bytes[i / 2] = (byte)(stored << 4);
                }
                else
                {
                    bytes[i / 2] |= stored;
                }
            }

            return bytes;
        }
    }

    /// <summary>
    /// Quantizes <paramref name="x"/> onto the int4 grid using the supplied affine parameters.
    /// </summary>
    /// <param name="x">Source float tensor.</param>
    /// <param name="scale">Quantization scale.</param>
    /// <param name="zeroPoint">Quantization zero-point.</param>
    /// <param name="symmetric">When true, the grid is signed [-7, 7]; otherwise unsigned [0, 15].</param>
    /// <returns>A new <see cref="Int4PackedTensor"/>.</returns>
    public static Int4PackedTensor FromFloat(Tensor x, float scale, long zeroPoint, bool symmetric)
    {
        ArgumentNullException.ThrowIfNull(x);
        var qMin = symmetric ? -7L : 0L;
        var qMax = symmetric ? 7L : 15L;

        using var divided = x.cpu().div(scale);
        using var rounded = divided.round();
        using var shifted = rounded.add(zeroPoint);
        using var clamped = shifted.clamp(qMin, qMax);
        using var asLong = clamped.to(ScalarType.Int64);

        var data = asLong.data<long>().ToArray();
        var ints = new sbyte[data.Length];
        for (var i = 0; i < data.Length; i++)
        {
            // Subtract zero point back so internal representation is signed in [-7, 7] or [-zp, 15-zp].
            ints[i] = (sbyte)(data[i] - zeroPoint);
        }

        return new Int4PackedTensor(ints, scale, zeroPoint, symmetric);
    }

    /// <summary>
    /// Reconstructs from a packed byte buffer produced by <see cref="PackedBytes"/>.
    /// </summary>
    /// <param name="packed">Packed byte buffer.</param>
    /// <param name="elementCount">Number of int4 elements encoded in <paramref name="packed"/>.</param>
    /// <param name="scale">Quantization scale used at packing time.</param>
    /// <param name="zeroPoint">Quantization zero-point used at packing time.</param>
    /// <param name="symmetric">Whether the original quantization was symmetric.</param>
    /// <returns>A reconstructed <see cref="Int4PackedTensor"/>.</returns>
    public static Int4PackedTensor FromPackedBytes(
        byte[] packed,
        int elementCount,
        float scale,
        long zeroPoint,
        bool symmetric)
    {
        ArgumentNullException.ThrowIfNull(packed);
        if (packed.Length != (elementCount + 1) / 2)
        {
            throw new ArgumentException(
                $"Expected {(elementCount + 1) / 2} bytes for {elementCount} int4 elements; got {packed.Length}.",
                nameof(packed));
        }

        var ints = new sbyte[elementCount];
        for (var i = 0; i < elementCount; i++)
        {
            var b = packed[i / 2];
            var nibble = ((i & 1) == 0) ? (byte)(b >> 4) : (byte)(b & 0x0F);
            sbyte signed;
            if (symmetric)
            {
                // Sign-extend from 4-bit
                signed = (nibble & 0x08) != 0 ? (sbyte)(nibble | unchecked((sbyte)0xF0)) : (sbyte)nibble;
            }
            else
            {
                signed = (sbyte)nibble;
            }

            ints[i] = signed;
        }

        return new Int4PackedTensor(ints, scale, zeroPoint, symmetric);
    }

    /// <summary>
    /// Dequantizes back to an FP32 tensor: equivalent to <c>int * scale</c> for the stored signed values.
    /// </summary>
    /// <returns>A new FP32 tensor with the dequantized values.</returns>
    public Tensor Dequantize()
    {
        var floats = new float[_ints.Length];
        for (var i = 0; i < _ints.Length; i++)
        {
            floats[i] = _ints[i] * _scale;
        }

        return tensor(floats);
    }
}
