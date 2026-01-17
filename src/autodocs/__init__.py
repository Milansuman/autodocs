from .parsers import PythonParser
from rich import print

def main() -> None:
    parser = PythonParser()
    with open("src/autodocs/parsers.py", "r") as f:
        parser.parse(f.read())
        #print(parser.get_constants())
        #print(parser.get_functions())
        print(parser.get_classes())

if __name__ == "__main__":
    main()