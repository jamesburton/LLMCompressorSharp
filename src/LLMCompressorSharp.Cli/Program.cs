// Copyright (c) James Burton. Licensed under the Apache-2.0 license.

using System.Reflection;
using static TorchSharp.torch;

namespace LLMCompressorSharp.Cli;

/// <summary>
/// Entry point for the <c>llmc</c> command-line tool.
/// </summary>
public static class Program
{
    /// <summary>
    /// Runs the CLI. Phase 0 implements <c>version</c> and <c>smoke</c> subcommands only;
    /// the full <c>compress</c> command lands in Phase 6.
    /// </summary>
    /// <param name="args">Command-line arguments.</param>
    /// <returns>Process exit code. 0 on success, non-zero on failure.</returns>
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] == "--help" || args[0] == "-h")
        {
            PrintHelp();
            return 0;
        }

        return args[0] switch
        {
            "version" => RunVersion(),
            "smoke" => RunSmoke(),
            _ => UnknownCommand(args[0]),
        };
    }

    private static int RunVersion()
    {
        var version = Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? "0.0.0-unknown";
        Console.WriteLine($"llmc {version}");
        return 0;
    }

    private static int RunSmoke()
    {
        Console.WriteLine("LLMCompressorSharp smoke test...");
        var t = zeros(3, 3);
        Console.WriteLine($"Created TorchSharp tensor shape=[{string.Join(",", t.shape)}] dtype={t.dtype}");
        t.Dispose();
        Console.WriteLine("OK");
        return 0;
    }

    private static int UnknownCommand(string name)
    {
        Console.Error.WriteLine($"Unknown command: {name}");
        PrintHelp();
        return 1;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("Usage: llmc <command>");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  version    Print the installed llmc version.");
        Console.WriteLine("  smoke      Run a minimal TorchSharp smoke test.");
    }
}
