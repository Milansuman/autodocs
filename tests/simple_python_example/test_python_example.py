from autodocs.parsers import PythonParser

parser = PythonParser()

with open("tests/simple_python_example/example.py", "r") as f:
    parser.parse(f.read())


def test_parse_module_variables() -> None:
    module_vars = parser.get_constants()

    assert len(module_vars) == 41
    assert module_vars[0]["name"] == "a" and module_vars[0]["value"] == "[1, 2, 3, 4]"
    assert module_vars[3]["name"] == "blfd" and module_vars[3]["type"] == "str"

def test_parse_module_functions() -> None:
    functions = parser.get_functions()

    assert len(functions) == 14
    assert functions[0]["name"] == "add" and functions[0]["parameters"] == ["x", "y"]
    assert ( functions[13]["name"] == "my_function" 
            and functions[13]["docstring"] == "Adds two numbers together." 
            and functions[13]["decorators"] == ["log_function"] )

def test_parse_module_classes() -> None:
    classes = parser.get_classes()

    assert len(classes) == 4
    assert classes[0]["name"] == "Human" and len(classes[0]["methods"]) == 8
    assert classes[1]["name"] == "Superhero" and "Human" in classes[1]["bases"]
    assert "classmethod" in classes[0]["methods"][3]["decorators"]
    