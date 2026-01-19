import os
from .parsers import PythonParser, Constant, Function, Class

def find_project_sources(extensions: list[str], path: str =".") -> list[str]:
    """Find all source files in the given path with the specified extensions.

    Args:
        extensions (list[str]): List of file extensions to look for.
        path (str, optional): The root directory to search. Defaults to the current directory.

    Returns:
        list[str]: List of file paths matching the specified extensions.
    """
    matched_files = []

    for root, dirs, files in os.walk(path):
        # Remove hidden directories from traversal
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith(tuple(extensions)):
                matched_files.append(os.path.join(root, file))

    return matched_files

def get_module_docstring(path: str, lang: str) -> str:
    """Extract the module-level docstring from the given file.

    Args:
        path (str): Path to the source file.
        lang (str): Programming language of the source file.

    Returns:
        str: The module-level docstring.
    """
    if lang == "python":
        parser = PythonParser()
        with open(path, "r") as file:
            parser.parse(file.read())
        return parser.get_module_docstring()
    else:
        raise ValueError(f"Unsupported language: {lang}")

def get_module_variables(path: str, lang: str, include_private: bool = False) -> list[Constant]:
    """Extract module-level variables from the given file.

    Args:
        path (str): Path to the source file.
        lang (str): Programming language of the source file.
        include_private (bool): Whether to include private variables (starting with _). Defaults to False.

    Returns:
        list[Constant]: List of module-level variables.
    """
    if lang == "python":
        parser = PythonParser()
        with open(path, "r") as file:
            parser.parse(file.read())
        constants = parser.get_constants()
        if not include_private:
            constants = [c for c in constants if not c['name'].startswith('_')]
        return constants
    else:
        raise ValueError(f"Unsupported language: {lang}")
    

def get_module_functions(path: str, lang: str, include_private: bool = False) -> list[Function]:
    """Extract module-level functions from the given file.

    Args:
        path (str): Path to the source file.
        lang (str): Programming language of the source file.
        include_private (bool): Whether to include private functions (starting with _). Defaults to False.

    Returns:
        list[Function]: List of module-level functions.
    """
    if lang == "python":
        parser = PythonParser()
        with open(path, "r") as file:
            parser.parse(file.read())
        functions = parser.get_functions()
        if not include_private:
            functions = [f for f in functions if not f['name'].startswith('_')]
        return functions
    else:
        raise ValueError(f"Unsupported language: {lang}")
    
def get_module_classes(path: str, lang: str, include_private: bool = False) -> list[Class]:
    """Extract module-level classes from the given file.

    Args:
        path (str): Path to the source file.
        lang (str): Programming language of the source file.
        include_private (bool): Whether to include private classes (starting with _). Defaults to False.

    Returns:
        list[Class]: List of module-level classes.
    """
    if lang == "python":
        parser = PythonParser()
        with open(path, "r") as file:
            parser.parse(file.read())
        classes = parser.get_classes()
        if not include_private:
            classes = [c for c in classes if not c['name'].startswith('_')]
        return classes
    else:
        raise ValueError(f"Unsupported language: {lang}")
    
def get_file_symbols(path: str, lang: str) -> str:
    """
    Get a summary of all symbols (variables, functions, classes) in the given file.
    Args:
        path (str): Path to the source file.
        lang (str): Programming language of the source file.
    """

    if lang == "python":
        parser = PythonParser()
        with open(path, "r") as file:
            parser.parse(file.read())
        constants = parser.get_constants()
        functions = parser.get_functions()
        classes = parser.get_classes()

        summary = f"File: {path}\n\n"

        if constants:
            summary += "Module-level Variables:\n"
            for const in constants:
                summary += f"- {const['name']}: {const['value']}\n"
            summary += "\n"

        if functions:
            summary += "Module-level Functions:\n"
            for func in functions:
                summary += f"- {func['name']}({', '.join(func['parameters'])})\n"
            summary += "\n"

        if classes:
            summary += "Module-level Classes:\n"
            for cls in classes:
                summary += f"- {cls['name']}\n"
            summary += "\n"

        return summary.strip()
    else:
        raise ValueError(f"Unsupported language: {lang}")
    
def get_specific_function(path: str, lang: str, function_name: str) -> Function | None:
    """Get a specific function by name from a file.

    Args:
        path (str): Path to the source file.
        lang (str): Programming language of the source file.
        function_name (str): Name of the function to retrieve.

    Returns:
        Function | None: The function if found, None otherwise.
    """
    functions = get_module_functions(path, lang, include_private=True)
    for func in functions:
        if func['name'] == function_name:
            return func
    return None

def get_specific_class(path: str, lang: str, class_name: str) -> Class | None:
    """Get a specific class by name from a file.

    Args:
        path (str): Path to the source file.
        lang (str): Programming language of the source file.
        class_name (str): Name of the class to retrieve.

    Returns:
        Class | None: The class if found, None otherwise.
    """
    classes = get_module_classes(path, lang, include_private=True)
    for cls in classes:
        if cls['name'] == class_name:
            return cls
    return None

def get_specific_variable(path: str, lang: str, variable_name: str) -> Constant | None:
    """Get a specific variable by name from a file.

    Args:
        path (str): Path to the source file.
        lang (str): Programming language of the source file.
        variable_name (str): Name of the variable to retrieve.

    Returns:
        Constant | None: The variable if found, None otherwise.
    """
    variables = get_module_variables(path, lang, include_private=True)
    for var in variables:
        if var['name'] == variable_name:
            return var
    return None

def save_documentation(path: str, documentation: str) -> None:
    """Save the generated documentation to a file.

    Args:
        path (str): Path to the output documentation file.
        documentation (str): The documentation content to save.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(path, "w") as file:
        file.write(documentation)
    
tools = {
    "find_project_sources": "Find all source files in the given path with the specified extensions.",
    "get_module_variables": "Extract module-level variables from the given file.",
    "get_module_functions": "Extract module-level functions from the given file.",
    "get_module_classes": "Extract module-level classes from the given file.",
    "get_file_symbols": "Get a summary of all symbols (variables, functions, classes) in the given file.",
    "save_documentation": "Save the generated documentation to a file.",
    "get_module_docstring": "Extract the module-level docstring from the given file."
}