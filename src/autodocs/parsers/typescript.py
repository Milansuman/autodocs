import tree_sitter
import tree_sitter_typescript
from typing import TypedDict
from dataclasses import dataclass

class Constant(TypedDict):
    name: str
    type: str | None
    value: str | None

class ReExportedSymbol(TypedDict):
    name: str

@dataclass
class TypescriptParser:
    """
    Class to parse typescript and tsx code for synbols relevant to generating documentation.
    """
    language_tsx: tree_sitter.Language = tree_sitter.Language(tree_sitter_typescript.language_tsx())
    language_typescript: tree_sitter.Language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    tsx_parser: tree_sitter.Parser = tree_sitter.Parser(language_tsx)
    typescript_parser: tree_sitter.Parser = tree_sitter.Parser(language_typescript)
    tree: tree_sitter.Tree | None = None

    def parse_ts(self, code: str) -> None:
        """Parse TypeScript code and build the syntax tree."""
        self.tree = self.typescript_parser.parse(bytes(code, "utf8"))

    def parse_tsx(self, code: str) -> None: 
        """Parse TSX code and build the syntax tree."""
        self.tree = self.tsx_parser.parse(bytes(code, "utf8"))

    def get_constants(self) -> list[Constant]:
        """
        Extract module-level constants from the parsed TypeScript/TSX code.

        Returns:
            list[Constant]: List of module-level constants.
        """
        if self.tree is None:
            raise ValueError("No syntax tree available. Please parse code first.")
 
        constants: list[Constant] = []
        root_node = self.tree.root_node

        query = """
        (program
            (export_statement
                declaration: (lexical_declaration
                    (variable_declarator
                        name: (identifier) @name
                        type: (type_annotation (_) @type)?
                        value: (_)? @value))))
        """

        constant_query = tree_sitter.Query(self.language_typescript, query)
        const_query_cursor = tree_sitter.QueryCursor(constant_query)

        for match in const_query_cursor.matches(root_node):
            match_node = match[1]
            constants.append({
                "name": match_node["name"][0].text.decode("utf8"),
                "type": match_node["type"][0].text.decode("utf8") if "type" in match_node else None,
                "value": match_node["value"][0].text.decode("utf8") if "value" in match_node else None
            })

        return constants
    
    def get_reexported_symbols(self) -> list[ReExportedSymbol]:
        """
        Extract re-exported symbols from the parsed TypeScript/TSX code.

        Returns:
            list[ReExportedSymbol]: List of re-exported symbols.
        """
        if self.tree is None:
            raise ValueError("No syntax tree available. Please parse code first.")

        reexported_symbols: list[ReExportedSymbol] = []
        root_node = self.tree.root_node

        # First, I get all the imports in the file
        # This is so that when I compare the symbols with the exported symbols, I can filter out only the re-exported ones
        # Getting re-exported symbols allows the documentation agent to understand how to write import statements in the generated documentation

        import_query = """
        (program
            (import_statement
                (import_clause
                    (named_imports
                        (import_specifier
                            name: (identifier) @name)*)
                source: (string) @source))
        """

        

