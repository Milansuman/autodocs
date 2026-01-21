import tree_sitter
import tree_sitter_python
from dataclasses import dataclass
from typing import TypedDict

class Constant(TypedDict):
    name: str
    type: str | None
    value: any

class Function(TypedDict):
    name: str
    parameters: list[str]
    return_type: str | None
    docstring: str | None
    decorators: list[str] | None

class Class(TypedDict):
    name: str
    bases: list[str]
    docstring: str | None
    methods: list[Function]
    fields: list[Constant]

@dataclass
class PythonParser:
    """
    Class to parse python code for synbols relevant to generating documentation.
    """

    language: tree_sitter.Language = tree_sitter.Language(tree_sitter_python.language())
    parser: tree_sitter.Parser = tree_sitter.Parser(language)
    tree: tree_sitter.Tree | None = None

    def parse(self, code: str) -> None:
        """
        Parse a python file and save the parse tree.
        
        Args:
            code: The contents of the python file to parse.
        """
        self.tree = self.parser.parse(bytes(code, "utf8"))

    def get_tree(self) -> tree_sitter.Tree:
        """
        Get the parse tree of the last parsed python file.
        
        Returns:
            The parse tree.
        
        Raises:
            ValueError: If no file has been parsed yet.
        """
        if self.tree is None:
            raise ValueError("No file has been parsed yet.")
        return self.tree
    
    def get_module_docstring(self) -> str | None:
        """
        Get the module-level docstring from the parsed python file.
        
        Returns:
            The module docstring, or None if not found.
        
        Raises:
            ValueError: If no file has been parsed yet.
        """
        if self.tree is None:
            raise ValueError("No file has been parsed yet.")
        
        QUERY = """
        (module
            (expression_statement
                (string) @docstring))
        """

        docstring_query = tree_sitter.Query(self.language, QUERY)
        doc_query_cursor = tree_sitter.QueryCursor(docstring_query)
        
        for match in doc_query_cursor.matches(self.tree.root_node):
            match_node = match[1]
            return match_node["docstring"][0].text.decode("utf8").strip('"""').strip("'''")
        
        return None
    
    def get_constants(self) -> list[Constant]:
        """
        Get all constant definitions from the parsed python file.
        
        Returns:
            A list of constant definitions.
        
        Raises:
            ValueError: If no file has been parsed yet.
        """
        if self.tree is None:
            raise ValueError("No file has been parsed yet.")
        
        constants: list[Constant] = []

        QUERY = """
        (module
            (expression_statement
                (assignment
                    left: (identifier) @name
                    type: (type)? @type
                    right: (_) @value)))
        """

        constant_query = tree_sitter.Query(self.language, QUERY)
        const_query_cursor = tree_sitter.QueryCursor(constant_query)
        
        for match in const_query_cursor.matches(self.tree.root_node):
            match_node = match[1]
            constants.append({
                "name": match_node["name"][0].text.decode("utf8"),
                "type": match_node["type"][0].text.decode("utf8") if "type" in match_node else None,
                "value": match_node["value"][0].text.decode("utf8"),
            })
        
        return constants
    
    def get_functions(self) -> list[Function]:
        """
        Get all module functions from the parsed python file.
        
        Returns:
            A list of Function definitions with their docstrings.
        
        Raises:
            ValueError: If no file has been parsed yet.
        """

        if self.tree is None:
            raise ValueError("No file has been parsed yet.")
        
        functions: list[Function] = []

        QUERY = """
        (module
            (function_definition
                name: (identifier) @name
                parameters: (parameters) @parameters
                return_type: (type)? @return_type
                body: (block
                    (expression_statement
                        (string) @docstring)?)))
        
        (module
            (decorated_definition
                (decorator (identifier) @decorator)*
                definition: (function_definition
                    name: (identifier) @name
                    parameters: (parameters) @parameters
                    return_type: (type)? @return_type
                    body: (block
                        (expression_statement
                            (string) @docstring)?))))
        """

        function_query = tree_sitter.Query(self.language, QUERY)
        func_query_cursor = tree_sitter.QueryCursor(function_query)
        
        for match in func_query_cursor.matches(self.tree.root_node):
            match_node = match[1]
            functions.append({
                "name": match_node["name"][0].text.decode("utf8"),
                "parameters": [
                    param.text.decode("utf8")
                    for param in match_node["parameters"][0].children
                    if param.type != "," and param.type != "(" and param.type != ")"
                ],
                "return_type": match_node["return_type"][0].text.decode("utf8") if "return_type" in match_node else None,
                "docstring": match_node["docstring"][0].text.decode("utf8").strip('"""').strip("'''") if "docstring" in match_node else None,
                "decorators": [
                    decorator.text.decode("utf8")
                    for decorator in match_node["decorator"]
                ] if "decorator" in match_node else None,
            })
        
        return functions
    
    def get_classes(self) -> list[Class]:
        """
        Get all class definitions from the parsed python file.
        
        Returns:
            A list of Class definitions with their docstrings and methods.
        
        Raises:
            ValueError: If no file has been parsed yet.
        """
        if self.tree is None:
            raise ValueError("No file has been parsed yet.")
        
        classes: list[Class] = []
        
        # Query to find all class definitions
        CLASS_QUERY = """
        (class_definition
            name: (identifier) @name
            superclasses: (argument_list)? @bases
            body: (block) @body) @class
        
        (decorated_definition
            definition: (class_definition
                name: (identifier) @name
                superclasses: (argument_list)? @bases
                body: (block) @body)) @class
        """
        
        # Query to find methods within a class body
        METHOD_QUERY = """
        (block
            (function_definition
                name: (identifier) @method.name
                parameters: (parameters) @method.params
                return_type: (type)? @method.return_type
                body: (block) @method.body) @method)
        
        (block
            (decorated_definition
                (decorator (_) @decorator)*
                definition: (function_definition
                    name: (identifier) @method.name
                    parameters: (parameters) @method.params
                    return_type: (type)? @method.return_type
                    body: (block) @method.body)) @method)
        """
        
        # Query to find class-level fields
        FIELD_QUERY = """
        (block
            (expression_statement
                (assignment
                    left: (_) @field.name
                    type: (type)? @type
                    right: (_) @field.value)) .)
        """
        
        class_query = tree_sitter.Query(self.language, CLASS_QUERY)
        method_query = tree_sitter.Query(self.language, METHOD_QUERY)
        field_query = tree_sitter.Query(self.language, FIELD_QUERY)
        
        class_cursor = tree_sitter.QueryCursor(class_query)
        
        for match in class_cursor.matches(self.tree.root_node):
            captures = match[1]
            
            # Get the class node
            class_node = captures["class"][0]
            if class_node.type == "decorated_definition":
                class_node = class_node.child_by_field_name("definition")
            
            name = captures["name"][0].text.decode("utf8")
            body_node = captures["body"][0]
            
            # Parse bases
            bases = []
            if "bases" in captures:
                bases_node = captures["bases"][0]
                for child in bases_node.named_children:
                    bases.append(child.text.decode("utf8"))
            
            # Parse docstring
            docstring = None
            first_stmt = body_node.named_child(0)
            if first_stmt and first_stmt.type == "expression_statement":
                first_expr = first_stmt.named_child(0)
                if first_expr and first_expr.type == "string":
                    docstring = first_expr.text.decode("utf8").strip('"""').strip("'''")
            
            # Parse fields using a query on the body node
            fields: list[Constant] = []
            field_cursor = tree_sitter.QueryCursor(field_query)
            for field_match in field_cursor.matches(body_node):
                field_captures = field_match[1]
                field_name_node = field_captures["field.name"][0]
                field_value_node = field_captures["field.value"][0]
                field_type_node = field_captures["type"][0] if "type" in field_captures else None
                
                if field_name_node.parent.parent == body_node:
                    field_info = {
                        "name": field_name_node.text.decode("utf8"),
                        "value": field_value_node.text.decode("utf8"),
                        "type": field_type_node.text.decode("utf8") if field_type_node else None,
                    }
                    fields.append(field_info)
                
            # Parse methods using a query on the body node
            methods: list[Function] = []
            method_cursor = tree_sitter.QueryCursor(method_query)
            for method_match in method_cursor.matches(body_node):
                method_captures = method_match[1]
                
                method_node = method_captures["method"][0]
                if method_node.type == "decorated_definition":
                    method_node = method_node.child_by_field_name("definition")
                
                method_info = {
                    "name": method_captures["method.name"][0].text.decode("utf8"),
                    "parameters": [
                        param.text.decode("utf8")
                        for param in method_captures["method.params"][0].named_children
                    ],
                    "return_type": (
                        method_captures["method.return_type"][0].text.decode("utf8")
                        if "method.return_type" in method_captures
                        else None
                    ),
                    "decorators": [
                        decorator.text.decode("utf8")
                        for decorator in method_captures.get("decorator", [])
                    ] if "decorator" in method_captures else None,
                }
                
                # Extract method docstring
                method_body = method_captures["method.body"][0]
                first_method_stmt = method_body.named_child(0)
                if first_method_stmt and first_method_stmt.type == "expression_statement":
                    maybe_docstring = first_method_stmt.named_child(0)
                    if maybe_docstring and maybe_docstring.type == "string":
                        method_info["docstring"] = maybe_docstring.text.decode("utf8").strip('"""').strip("'''")
                    else:
                        method_info["docstring"] = None
                else:
                    method_info["docstring"] = None
                
                methods.append(method_info)
            
            classes.append({
                "name": name,
                "bases": bases,
                "docstring": docstring,
                "methods": methods,
                "fields": fields
            })
        
        return classes