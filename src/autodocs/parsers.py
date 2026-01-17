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
        
        :param code: The contents of the python file to parse.
        :type code: str
        """
        self.tree = self.parser.parse(bytes(code, "utf8"))

    def get_tree(self) -> tree_sitter.Tree:
        """
        Get the parse tree of the last parsed python file.
        
        :return: The parse tree.
        :rtype: tree_sitter.Tree
        :raises ValueError: If no file has been parsed yet.
        """
        if self.tree is None:
            raise ValueError("No file has been parsed yet.")
        return self.tree
    
    def get_constants(self) -> list[Constant]:
        """
        Get all constant definitions from the parsed python file.
        
        :return: A list of constant definitions.
        :rtype: list[Constant]
        :raises ValueError: If no file has been parsed yet.
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
        
        :return: A list of Function definitions with their docstrings.
        :rtype: list[Function]
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
            })
        
        return functions
    
    def get_classes(self) -> list[Class]:
        """
        Get all class definitions from the parsed python file.

        :return: A list of Class definitions with their docstrings and methods.
        """

        if self.tree is None:
            raise ValueError("No file has been parsed yet.")
        
        classes: list[Class] = []

        QUERY = """
        (module
            (class_definition
                name: (identifier) @name
                superclasses: (argument_list)? @bases
                body: (block
                    (expression_statement
                        (string) @docstring)?
                    (expression_statement
                        (assignment))?* @fields
                    (function_definition)?* @methods)
                    ))

        (module
            (decorated_definition
                definition: (class_definition
                    name: (identifier) @name
                    superclasses: (argument_list)? @bases
                    body: (block
                        (expression_statement
                            (string) @docstring)?
                        (expression_statement
                            (assignment))?* @fields
                        (function_definition)?* @methods)
                        )))
        """

        class_query = tree_sitter.Query(self.language, QUERY)
        class_query_cursor = tree_sitter.QueryCursor(class_query)
        
        for match in class_query_cursor.matches(self.tree.root_node):
            match_node = match[1]
            
            fields: list[Constant] = []
            if "fields" in match_node:
                for field in match_node["fields"]:
                    assignment_node = field.named_child(0)
                    field_info = {
                        "name": assignment_node.child(0).text.decode("utf8"),
                        "type": assignment_node.child(2).text.decode("utf8"),
                        "value": assignment_node.child(4).text.decode("utf8") if assignment_node.child_count > 4 else None,
                    }
                    fields.append(field_info)
            methods: list[Function] = []
            if "methods" in match_node:
                for method in match_node["methods"]:
                    method_info = {
                        "name": method.child_by_field_name("name").text.decode("utf8"),
                        "parameters": [
                            param.text.decode("utf8")
                            for param in method.child_by_field_name("parameters").children
                            if param.type != "," and param.type != "(" and param.type != ")"
                        ],
                        "return_type": method.child_by_field_name("return_type").text.decode("utf8") if method.child_by_field_name("return_type") else None
                    }

                    maybe_docstring = method.child_by_field_name("body").named_child(0).named_child(0)
                    if maybe_docstring.type == "string":
                        method_info["docstring"] = maybe_docstring.text.decode("utf8").strip('"""').strip("'''")
                    else:
                        method_info["docstring"] = None
                    
                    methods.append(method_info)

            classes.append({
                "name": match_node["name"][0].text.decode("utf8"),
                "bases": [
                    base.text.decode("utf8")
                    for base in match_node["bases"][0].children
                    if base.type != "," and base.type != "(" and base.type != ")"
                ] if "bases" in match_node else [],
                "docstring": match_node["docstring"][0].text.decode("utf8").strip('"""').strip("'''") if "docstring" in match_node else None,
                "methods": methods,
                "fields": fields
            })
        
        return classes