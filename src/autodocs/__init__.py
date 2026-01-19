from .parsers import PythonParser
from .agent import generate_documentation
from rich import print

def main() -> None:
    """Main entry point for autodocs CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate documentation for your codebase")
    parser.add_argument("--path", default=".", help="Path to the project to document")
    parser.add_argument("--output", default="docs", help="Output directory for documentation")
    parser.add_argument("--extensions", nargs="+", default=[".py"], help="File extensions to process")
    parser.add_argument("--topics", nargs="+", help="Topics to organize documentation by (e.g., 'parsers' 'tools' 'agent')")
    
    args = parser.parse_args()
    
    print(f"[bold blue]Generating documentation for: {args.path}[/bold blue]")
    if args.topics:
        print(f"[dim]Organizing by topics: {', '.join(args.topics)}[/dim]")
    
    result = generate_documentation(
        project_path=args.path,
        codebase_extensions=args.extensions,
        output_dir=args.output,
        topics=args.topics
    )
    
    print(f"[bold green]✓ Documentation generated in: {args.output}[/bold green]")
    print(f"[dim]Processed {len(result['file_paths'])} files[/dim]")

if __name__ == "__main__":
    main()
