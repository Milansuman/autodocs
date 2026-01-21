from autodocs.parsers.typescript import TypescriptParser

ts = TypescriptParser()
with open("tests/simple_typescript_example/example.ts", "r") as f:
    ts.parse_ts(f.read())

print(ts.get_constants())