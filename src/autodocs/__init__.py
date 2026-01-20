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
    parser.add_argument("--extrapolate", action="store_true", help="Automatically extrapolate topics from the codebase")
    
    args = parser.parse_args()

    generate_documentation(
        project_root=args.path,
        output_dir=args.output,
        extensions=args.extensions,
        topics=args.topics,
        extrapolate=args.extrapolate
    )

if __name__ == "__main__":
    main()
