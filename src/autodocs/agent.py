from typing import TypedDict, Annotated
from langchain.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import json
from rich import print

from .tools import (
    find_config_files, 
    find_project_sources,
    get_module_docstring,
    get_module_variables,
    get_module_functions,
    get_module_classes,
    save_documentation
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_STRING = os.getenv("GROQ_MODEL_STRING", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

llm = ChatGroq(
    model=GROQ_MODEL_STRING,
    api_key=GROQ_API_KEY
)

class AgentContext(TypedDict):
    files: dict[str, str] | None
    symbols: dict[str, list[dict]] | None
    config: str | None
    topics: list[str] | None
    project_root: str
    extensions: list[str]
    output_dir: str
    extrapolate: bool

class TopicSortedContext(TypedDict):
    topic_files: dict[str, list[dict]]

class TopicCategorizationOutput(BaseModel):
    topic_files: dict[str, list[str]]

class TopicExtrapolationOutput(BaseModel):
    topics: list[str]

def read_config_node(state: AgentContext) -> AgentContext:
    project_root = state["project_root"]
    extensions = state["extensions"]

    config = ""
    
    if ".py" in extensions:
        config_path = os.path.join(project_root, "pyproject.toml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = f.read()
    elif ".ts" in extensions or ".tsx" in extensions:
        config_path = os.path.join(project_root, "package.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = f.read()

    return {
        "config": config
    }

def read_project_files_node(state: AgentContext) -> AgentContext:
    project_root = state["project_root"]
    extensions = state["extensions"]
    
    source_files = find_project_sources(extensions, project_root)
    
    files = {}
    for filepath in source_files:
        try:
            # Determine parser type based on file extension
            parser_lang = "python" if filepath.endswith(".py") else "typescript"
            docstring = get_module_docstring(filepath, parser_lang)
            files[filepath] = docstring
        except Exception:
            files[filepath] = ""
    
    return {
        "files": files
    }

def extrapolate_topics_node(state: AgentContext) -> AgentContext:
    topics = state.get("topics")
    files = state.get("files", {})

    system_prompt = """You are an agent that suggests relevant documentation topics based on the provided file descriptions and given topics. If no topics are provided, suggest a list of relevant topics based on the file descriptions.
    
    GUIDELINES:
    - If topics are provided, suggest additional relevant topics that complement the existing ones.
    - If no topics are provided, suggest a list of relevant topics based on the file descriptions.
    - Ensure the topics are broad enough to cover multiple files but specific enough to be meaningful.
    - Ensure the topics are in the form of phrases suitable for getting started guides.
    - Return the topics in a JSON array format.

    EXAMPLE OUTPUT:
    {
        topics: [
            "Getting Started with Parsers",
            "Tools QuitStart",
            "Agent Configuration and Setup"
        ]
    }
    """

    human_prompt = f"""FILES:
    {"\n".join([f"{path}: {doc}" for path, doc in files.items()])}
    EXISTING TOPICS:
    {', '.join(topics) if topics else 'none'}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)
    try:
        response_data = json.loads(response.content)
        TopicExtrapolationOutput(**response_data)
        return {
            "topics": response_data["topics"]
        }
    except Exception as e:
        raise ValueError(f"Failed to parse response: {response.content}") from e

    
def categorize_files_node(state: AgentContext) -> TopicSortedContext:
    topics = state.get("topics", [])
    files = state.get("files", {})

    system_prompt = """
You are an agent that categorizes source code files into topics based on their description. You will be given a list of topics and a list of files with their docstrings. Your task is to assign each file to the most relevant topic based on its docstring. If a file does not clearly belong to any topic, do not include it in the output.

EXAMPLE OUTPUT:
{
    "topic_files": {
        "topic1": ["path/to/file1.py", "path/to/file2.py"],
        "topic2": ["path/to/file3.py"]
    }
}
"""

    human_prompt = f"""
TOPICS:
{', '.join(topics) if topics else 'all'}
FILES AND DESCRIPTIONS:
{"\n".join([f"{path}: {doc}" for path, doc in files.items()])}
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)

    try:
        response_data = json.loads(response.content)
        print(response_data)
        TopicCategorizationOutput(**response_data)
        return response_data
    except Exception as e:
        raise ValueError(f"Failed to parse response: {response.content}") from e
    
def generate_docs_node(state: TopicSortedContext):
    topic_files = state["topic_files"]
    output_dir = state.get("output_dir", "docs")
    config = state.get("config", "")

    system_prompt = """You are an agent that generates documentation for the given symbols by the user. A topic and it's associated symbols will be provided with their file paths. Generate clear and concise documentation in the style of a quick start guide.
    
    DOCUMENTATION GUIDELINES:
    - The documentation should be in the form of a quick start guide relevant to the topic.
    - Use clear headings and subheadings.
    - Include code examples where applicable.
    - Keep the language simple and accessible.
    - Do NOT generate reference documentation or API docs listing all the options with their types.
    - Always use markdown to generate your documentation.

    INPUT FORMAT:
    TOPIC: <topic_name>

    CONFIG FILE CONTENTS:
    <config file contents>

    - <file_path_1>
    CONSTANTS:
    <json data describing the file constants>

    FUNCTIONS:
    <json data describing the file functions>

    CLASSES:
    <json data describing the file classes>

    - <file_path_2>
    CONSTANTS:
    <json data describing the file constants>

    FUNCTIONS:
    <json data describing the file functions>

    CLASSES:
    <json data describing the file classes>
    """

    for topic, files in topic_files.items():
        print(f"[bold underline]Topic: {topic}[/bold underline]")

        human_prompt = f"""TOPIC: {topic}\n\nCONFIG FILE CONTENTS:
        {config}\n\n"""

        for file in files:
            print(f"- {file}")
            human_prompt += f"- {file}\n"

            constants = get_module_variables(file, "python")
            functions = get_module_functions(file, "python")
            classes = get_module_classes(file, "python")

            human_prompt += f"""CONSTANTS:
{"\n".join([json.dumps(const) for const in constants])}

FUNCTIONS:
{"\n".join([json.dumps(func) for func in functions])}

CLASSES:
{"\n".join([json.dumps(cls) for cls in classes])}
"""
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = llm.invoke(messages)
        print(f"[bold green]Generated Documentation for Topic: {topic}[/bold green]")
        save_documentation(f"{output_dir}/{topic}.md", response.content)

def generate_documentation(
    project_root: str = ".",
    output_dir: str = "docs",
    extensions: list[str] | None = None,
    topics: list[str] | None = None,
    extrapolate: bool = True
):
    """Main function to generate documentation for a project."""
    
    # Handle None initialization
    if extensions is None:
        extensions = [".py"]
    if topics is None:
        topics = []

    builder = StateGraph(AgentContext)
    builder.add_node("read_config", read_config_node)
    builder.add_node("read_project_files", read_project_files_node)
    builder.add_node("extrapolate_topics", extrapolate_topics_node)
    builder.add_node("categorize_files", categorize_files_node)
    builder.add_node("generate_docs", generate_docs_node)

    builder.add_edge(START, "read_config")
    builder.add_edge("read_config", "read_project_files")
    builder.add_edge("read_project_files", "extrapolate_topics")
    builder.add_edge("extrapolate_topics", "categorize_files")
    builder.add_edge("categorize_files", "generate_docs")
    builder.add_edge("generate_docs", END)

    agent = builder.compile()

    agent.invoke({
        "project_root": project_root,
        "output_dir": output_dir,
        "extensions": extensions,
        "topics": topics,
        "extrapolate": extrapolate,
        "files": None,
        "symbols": None,
        "config": None
    })