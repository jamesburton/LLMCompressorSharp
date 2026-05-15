# C# 14 — New Language Features (.NET 10)

C# 14 ships with .NET 10 and requires the .NET 10 SDK (or Visual Studio 2026+).

---

## Extension Members

The new `extension` block syntax allows declaring extension **properties** (instance and static) and **static extension methods** alongside the existing extension method syntax. User-defined operators can be declared as static extension methods.

```csharp
public static class Enumerable
{
    extension<TSource>(IEnumerable<TSource> source)
    {
        // Instance extension property
        public bool IsEmpty => !source.Any();

        // Instance extension method
        public IEnumerable<TSource> Where(Func<TSource, bool> predicate) { ... }
    }

    extension<TSource>(IEnumerable<TSource>)
    {
        // Static extension property
        public static IEnumerable<TSource> Identity => Enumerable.Empty<TSource>();

        // Extension operator
        public static IEnumerable<TSource> operator +(
            IEnumerable<TSource> left, IEnumerable<TSource> right)
            => left.Concat(right);
    }
}
```

**ML relevance:** Enables `tensor1 + tensor2` syntax when `T` implements `IAdditionOperators<...>`. The `System.Numerics.Tensors` library uses this for `Tensor<T>` arithmetic operators.

---

## `field` Keyword (Field-Backed Properties)

The `field` contextual keyword accesses the compiler-generated backing field inside a property accessor — eliminating explicit backing field declarations for simple validation:

```csharp
public string Name
{
    get;
    set => field = value ?? throw new ArgumentNullException(nameof(value));
}

public float Scale
{
    get;
    set
    {
        if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value));
        field = value;
    }
}
```

---

## Implicit `Span<T>` / `ReadOnlySpan<T>` Conversions

First-class language support for implicit conversions between `Span<T>`, `ReadOnlySpan<T>`, and `T[]`. Span types can now be extension method receivers, participate in generic type inference, and compose with other conversions.

**Before:** `AsSpan()` calls and explicit casts required.  
**After:** Many conversions happen implicitly, reducing noise in numeric code.

---

## Null-Conditional Assignment

The `?.` and `?[]` operators can now appear on the **left-hand side** of assignment and compound assignment (`+=`, `-=`, etc.):

```csharp
customer?.Order = GetCurrentOrder();
list?.Add(item);
observer?.Scale *= newValue;
```

---

## `nameof` with Unbound Generics

`nameof(List<>)` is now valid and returns `"List"`. Previously required `nameof(List<int>)`.

---

## Lambda Parameter Modifiers Without Types

`ref`, `in`, `out`, `scoped`, `ref readonly` can now be applied to lambda parameters without specifying the type:

```csharp
TryParse<int> parse = (text, out result) => int.TryParse(text, out result);
```

---

## Partial Constructors and Partial Events

Instance constructors and events can now be declared as `partial` members, requiring exactly one defining declaration and one implementing declaration. Useful for source generators that split generated and hand-written constructor logic.

```csharp
// Generated partial
public partial class QuantizationConfig
{
    public partial QuantizationConfig(string scheme);
}

// Hand-written partial
public partial class QuantizationConfig
{
    public partial QuantizationConfig(string scheme)
    {
        Scheme = scheme ?? throw new ArgumentNullException(nameof(scheme));
        // ... additional init
    }
}
```

---

## User-Defined Compound Assignment Operators

Custom `+=`, `-=`, `*=`, `/=` and `++`, `--` operators can now be defined:

```csharp
public static MyTensor operator +=(MyTensor left, MyTensor right)
{
    // in-place add
    return left;
}
```

---

## Preprocessor `#:` Directives (File-Based Apps)

Not strictly a language feature, but new C# syntax: `#:` directives at the top of a `.cs` file configure file-based apps (no `.csproj`):

```csharp
#:package TorchSharp-cpu@0.107.0
#:property PublishAot=false

using TorchSharp;
using static TorchSharp.torch;

var t = rand(3, 4);
Console.WriteLine(t);
```

See [sdk-tooling.md](./sdk-tooling.md) for full details.
