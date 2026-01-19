from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import operator
import os
from dotenv import load_dotenv
from .tools import (
    find_project_sources,
    get_file_symbols,
    get_module_docstring,
    get_module_variables,
    get_module_functions,
    get_module_classes,
    get_specific_function,
    get_specific_class,
    get_specific_variable,
    save_documentation,
    find_config_files,
    read_file
)

class AgentContext(TypedDict):
    project_path: str
    file_paths: list[str]
    current_file_index: int
    file_symbols: dict[str, str]
    file_docstrings: dict[str, str]  # Module-level docstrings
    file_variables: dict[str, list]  # Module-level variables
    file_functions: dict[str, list]  # Module-level functions
    file_classes: dict[str, list]  # Module-level classes
    documentation: dict[str, str]
    file_topics: dict[str, str]  # Maps file path to topic
    file_relevance: dict[str, bool]  # Maps file path to whether it's relevant to its topic
    doc_filenames: dict[str, str]  # Maps file path to descriptive doc filename
    messages: Annotated[list, operator.add]
    codebase_extensions: list[str]
    output_dir: str
    topics: list[str]  # Optional list of topics to organize by
    package_metadata: dict  # Package information from pyproject.toml or package.json

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_STRING = os.getenv("GROQ_MODEL_STRING")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

if not GROQ_MODEL_STRING:
    GROQ_MODEL_STRING = "openai/gpt-oss-120b"

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL_STRING
)

def extract_metadata_node(state: AgentContext) -> AgentContext:
    """Extract package metadata by having the LLM analyze config files."""
    project_path = state.get("project_path", ".")
    
    print("\n📦 Searching for package metadata...")
    
    # Find potential config files
    config_files = find_config_files(project_path)
    
    if not config_files:
        print("   No config files found")
        return {
            "package_metadata": {},
            "messages": [HumanMessage(content="No package metadata found")]
        }
    
    print(f"   Found {len(config_files)} potential config files")
    for cf in config_files:
        print(f"   - {os.path.basename(cf)}")
    
    # Read config files and let LLM decide what's relevant
    config_contents = {}
    for config_file in config_files[:5]:  # Limit to first 5 to avoid token overload
        try:
            content = read_file(config_file)
            # Limit content size
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            config_contents[os.path.basename(config_file)] = content
        except Exception as e:
            print(f"   ⚠️  Could not read {config_file}: {e}")
    
    if not config_contents:
        print("   Could not read any config files")
        return {
            "package_metadata": {},
            "messages": [HumanMessage(content="Could not read config files")]
        }
    
    # Use LLM to extract metadata
    print("   🤖 Using LLM to extract metadata...")
    
    files_text = "\n\n".join([f"=== {filename} ===\n{content}" for filename, content in config_contents.items()])
    
    prompt = f"""Analyze these project configuration files and extract package metadata.

{files_text}

Extract the following information if available:
- Package name
- Version
- Description
- Author/maintainer
- License

Respond in this exact format (use 'Unknown' if not found):
Name: <name>
Version: <version>
Description: <description>
Author: <author>
License: <license>"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Parse response
    package_metadata = {
        "name": None,
        "version": None,
        "description": None,
        "author": None,
        "license": None
    }
    
    for line in response_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            if value and value.lower() not in ['unknown', 'not found', 'n/a']:
                package_metadata[key] = value
    
    # Display what was found
    if package_metadata.get("name"):
        print(f"   ✅ Package: {package_metadata.get('name')} v{package_metadata.get('version', 'unknown')}")
        if package_metadata.get("description"):
            print(f"   Description: {package_metadata.get('description')[:100]}..." if len(package_metadata.get('description', '')) > 100 else f"   Description: {package_metadata.get('description')}")
    else:
        print("   ⚠️  Could not extract package name")
    
    return {
        "package_metadata": package_metadata,
        "messages": [HumanMessage(content="Extracted package metadata")]
    }

def find_sources_node(state: AgentContext) -> AgentContext:
    """Find all source files in the project."""
    project_path = state.get("project_path", ".")
    extensions = state.get("codebase_extensions", [".py"])
    
    print(f"\n🔍 Scanning project at: {project_path}")
    print(f"   Looking for files with extensions: {', '.join(extensions)}")
    
    file_paths = find_project_sources(extensions, project_path)
    print(f"\n✅ Found {len(file_paths)} source files to document\n")
    
    return {
        "file_paths": file_paths,
        "current_file_index": 0,
        "file_symbols": {},
        "documentation": {},
        "messages": [HumanMessage(content=f"Found {len(file_paths)} source files to document.")]
    }

def analyze_file_node(state: AgentContext) -> AgentContext:
    """Analyze the current file and extract symbols."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    file_symbols = state.get("file_symbols", {})
    file_docstrings = state.get("file_docstrings", {})
    file_variables = state.get("file_variables", {})
    file_functions = state.get("file_functions", {})
    file_classes = state.get("file_classes", {})
    
    if current_index >= len(file_paths):
        return {"messages": [HumanMessage(content="All files analyzed.")]}
    
    current_file = file_paths[current_index]
    
    print(f"\n[{current_index + 1}/{len(file_paths)}] 📄 Analyzing: {current_file}")
    
    # Determine language based on extension
    lang = "python" if current_file.endswith(".py") else "unknown"
    
    if lang == "unknown":
        print(f"   ⚠️  Skipping unsupported file type")
        return {
            "current_file_index": current_index + 1,
            "messages": [HumanMessage(content=f"Skipping unsupported file: {current_file}")]
        }
    
    # Get summary of symbols
    symbols = get_file_symbols(current_file, lang)
    file_symbols[current_file] = symbols
    
    # Get module docstring
    docstring = get_module_docstring(current_file, lang)
    file_docstrings[current_file] = docstring
    
    # First, get all symbols to see what's available (exclude private symbols)
    all_variables = get_module_variables(current_file, lang, include_private=False)
    all_functions = get_module_functions(current_file, lang, include_private=False)
    all_classes = get_module_classes(current_file, lang, include_private=False)
    
    # Now use specific tools to selectively fetch important symbols
    # For now, we'll fetch all public symbols, but this structure allows
    # for more selective fetching in the future (e.g., only exported symbols)
    selected_variables = []
    for var in all_variables:
        # Use specific tool to get detailed info
        specific_var = get_specific_variable(current_file, lang, var['name'])
        if specific_var:
            selected_variables.append(specific_var)
    
    selected_functions = []
    for func in all_functions:
        # Use specific tool to get detailed info
        specific_func = get_specific_function(current_file, lang, func['name'])
        if specific_func:
            selected_functions.append(specific_func)
    
    selected_classes = []
    for cls in all_classes:
        # Use specific tool to get detailed info
        specific_cls = get_specific_class(current_file, lang, cls['name'])
        if specific_cls:
            selected_classes.append(specific_cls)
    
    file_variables[current_file] = selected_variables
    file_functions[current_file] = selected_functions
    file_classes[current_file] = selected_classes
    
    # Print summary of what was found
    print(f"   Found: {len(selected_variables)} variables, {len(selected_functions)} functions, {len(selected_classes)} classes")
    
    return {
        "file_symbols": file_symbols,
        "file_docstrings": file_docstrings,
        "file_variables": file_variables,
        "file_functions": file_functions,
        "file_classes": file_classes,
        "messages": [HumanMessage(content=f"Analyzed file: {current_file} using specific symbol tools")]
    }

def categorize_file_node(state: AgentContext) -> AgentContext:
    """Categorize the current file into a topic and check relevance."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    file_symbols = state["file_symbols"]
    file_topics = state.get("file_topics", {})
    file_relevance = state.get("file_relevance", {})
    topics = state.get("topics", [])
    
    if current_index >= len(file_paths):
        return {"messages": [HumanMessage(content="All files categorized.")]}
    
    current_file = file_paths[current_index]
    symbols = file_symbols.get(current_file, "")
    
    if not symbols:
        file_topics[current_file] = "general"
        file_relevance[current_file] = False
        print(f"   ⏭️  Skipping - no symbols found")
        return {
            "file_topics": file_topics,
            "file_relevance": file_relevance,
            "messages": [HumanMessage(content=f"Skipping {current_file} - no symbols")]
        }
    
    # If topics are provided, categorize the file and check relevance
    if topics:
        categorization_prompt = f"""You are categorizing code files into topics and determining relevance.

Available topics: {', '.join(topics)}

File symbols:
{symbols}

Based on the symbols, determine:
1. The MOST appropriate topic from the list above
2. Whether this file contains symbols that would be USEFUL to document for users interested in that topic

Consider a file relevant only if it contains public APIs, classes, or functions that users would directly interact with.
Internal utilities, private helpers, or files with only implementation details should be marked as not relevant.

Respond in this exact format:
Topic: <topic_name>
Relevant: <yes/no>"""
        
        messages = [HumanMessage(content=categorization_prompt)]
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse response
        topic = "general"
        is_relevant = False
        
        for line in response_text.split('\n'):
            if line.startswith('Topic:'):
                topic = line.replace('Topic:', '').strip()
            elif line.startswith('Relevant:'):
                relevance_str = line.replace('Relevant:', '').strip().lower()
                is_relevant = relevance_str in ['yes', 'true']
        
        # Validate topic is in the list
        if topic not in topics:
            topic = topics[0] if topics else "general"
    else:
        # If no topics specified, mark all files with symbols as relevant
        topic = "general"
        is_relevant = True
    
    file_topics[current_file] = topic
    file_relevance[current_file] = is_relevant
    
    status = "relevant" if is_relevant else "not relevant for library users"
    print(f"   📂 Topic: {topic} | {'✅ Relevant' if is_relevant else '⏭️  Not relevant'}")
    
    return {
        "file_topics": file_topics,
        "file_relevance": file_relevance,
        "messages": [HumanMessage(content=f"Categorized {current_file} as: {topic} ({status})")]
    }

def generate_documentation_node(state: AgentContext) -> AgentContext:
    """Generate documentation for the current file using LLM."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    file_symbols = state["file_symbols"]
    file_docstrings = state.get("file_docstrings", {})
    file_variables = state.get("file_variables", {})
    file_functions = state.get("file_functions", {})
    file_classes = state.get("file_classes", {})
    file_relevance = state.get("file_relevance", {})
    documentation = state.get("documentation", {})
    file_topics = state.get("file_topics", {})
    doc_filenames = state.get("doc_filenames", {})
    
    if current_index >= len(file_paths):
        return {"messages": [HumanMessage(content="All documentation generated.")]}
    
    current_file = file_paths[current_index]
    symbols = file_symbols.get(current_file, "")
    topic = file_topics.get(current_file, "general")
    docstring = file_docstrings.get(current_file, "")
    variables = file_variables.get(current_file, [])
    functions = file_functions.get(current_file, [])
    classes = file_classes.get(current_file, [])
    is_relevant = file_relevance.get(current_file, True)
    
    # Skip if not relevant to the topic
    if not is_relevant:
        print(f"   ⏭️  Skipping documentation generation - not relevant")
        return {
            "current_file_index": current_index + 1,
            "messages": [HumanMessage(content=f"Skipping {current_file} - not relevant to library users")]
        }
    
    # Skip if no public symbols found
    if not symbols or (not variables and not functions and not classes):
        print(f"   ⏭️  Skipping documentation generation - no public symbols")
        return {
            "current_file_index": current_index + 1,
            "messages": [HumanMessage(content=f"No public symbols found for: {current_file}")]
        }
    
    # Build detailed context for the LLM (without mentioning file paths)
    detailed_context = f"""## Module Overview:
{docstring if docstring else 'No module docstring'}

## Available Functionality:
"""
    
    if variables:
        detailed_context += "\nConstants/Variables:\n"
        for var in variables:
            detailed_context += f"- **{var['name']}**: {var['value']}\n"
    
    if functions:
        detailed_context += "\nFunctions:\n"
        for func in functions:
            detailed_context += f"\n### {func['name']}({', '.join(func['parameters'])})\n"
            if func.get('docstring'):
                detailed_context += f"Docstring: {func['docstring']}\n"
    
    if classes:
        detailed_context += "\nClasses:\n"
        for cls in classes:
            detailed_context += f"\n### {cls['name']}"
            if cls.get('bases'):
                detailed_context += f" (extends: {', '.join(cls['bases'])})"
            detailed_context += "\n"
            if cls.get('docstring'):
                detailed_context += f"Docstring: {cls['docstring']}\n"
            if cls.get('methods'):
                detailed_context += "Public Methods:\n"
                for method in cls['methods']:
                    if not method['name'].startswith('_') or method['name'] in ['__init__', '__str__', '__repr__']:
                        detailed_context += f"  - {method['name']}({', '.join(method['parameters'])})\n"
                        if method.get('docstring'):
                            detailed_context += f"    {method['docstring']}\n"
    
    # Add package metadata context if available
    package_metadata = state.get("package_metadata", {})
    package_context = ""
    if package_metadata.get("name"):
        package_context = f"\n## Package Information:\n"
        package_context += f"Package: {package_metadata.get('name')}\n"
        if package_metadata.get('version'):
            package_context += f"Version: {package_metadata.get('version')}\n"
        if package_metadata.get('description'):
            package_context += f"Description: {package_metadata.get('description')}\n"
    
    print(f"   🤖 Generating documentation with LLM...")
    
    # Generate documentation using LLM
    system_prompt = """You are a technical documentation writer creating quickstart guides for library users.

Your documentation should:
1. Start with a brief, descriptive title (not the filename)
2. Focus ONLY on what library users need to know - not implementation details
3. Provide practical, working code examples showing how to use the functionality
4. Be concise and scannable
5. Don't mention file paths, internal structure, or implementation details
6. Don't document individual classes/functions - show how to accomplish tasks
7. Only document what exists - don't invent features
8. Assume the reader is using this library, not developing it

Format:
- Descriptive title reflecting the functionality (not filename)
- One-sentence overview of what this helps users accomplish
- 2-3 practical code examples showing common use cases
- Keep it short and actionable

DO NOT:
- Mention file paths or module structure
- Include internal implementation details
- Write API reference documentation
- Discuss how the code works internally"""

    user_prompt = f"""Create a quickstart guide for library users. Focus ONLY on how to use these features from a user's perspective.
{package_context}
{detailed_context}

Topic: {topic}

IMPORTANT:
- Generate a descriptive title based on the functionality (NOT the filename)
- Only document what's shown above
- Focus on HOW TO USE these features as a library user
- Show practical examples
- Don't mention file paths, internal structure, or implementation details
- Keep it concise and user-focused
- Don't use placeholder names for the library name. Use the actual package name.

Also provide a suggested filename for this documentation (short, kebab-case, descriptive, ending in .md)
Format: Filename: <name>.md"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content
    
    # Extract suggested filename
    doc_filename = None
    lines = response_text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('Filename:'):
            doc_filename = line.replace('Filename:', '').strip()
            # Remove this line from the documentation
            lines.pop(i)
            response_text = '\n'.join(lines)
            break
    
    # If no filename suggested, generate one from the topic and current file
    if not doc_filename:
        base = os.path.splitext(os.path.basename(current_file))[0]
        doc_filename = f"{topic}-{base}.md".replace('_', '-')
    
    doc_filenames[current_file] = doc_filename
    documentation[current_file] = response_text
    
    print(f"   ✅ Generated documentation: {doc_filename}")
    
    return {
        "documentation": documentation,
        "doc_filenames": doc_filenames,
        "messages": [HumanMessage(content=f"Generated documentation: {doc_filename}")]
    }

def save_documentation_node(state: AgentContext) -> AgentContext:
    """Save the documentation for the current file."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    documentation = state["documentation"]
    file_topics = state.get("file_topics", {})
    doc_filenames = state.get("doc_filenames", {})
    output_dir = state.get("output_dir", "docs")
    
    if current_index >= len(file_paths):
        return {"messages": [HumanMessage(content="All documentation saved.")]}
    
    current_file = file_paths[current_index]
    doc_content = documentation.get(current_file, "")
    topic = file_topics.get(current_file, "general")
    doc_filename = doc_filenames.get(current_file)
    
    if not doc_content:
        return {
            "current_file_index": current_index + 1,
            "messages": [HumanMessage(content=f"No documentation to save for: {current_file}")]
        }
    
    # Use the descriptive filename generated by the LLM
    if not doc_filename:
        # Fallback if no filename was generated
        base = os.path.splitext(os.path.basename(current_file))[0]
        doc_filename = f"{topic}-{base}.md".replace('_', '-')
    
    output_path = os.path.join(output_dir, topic, doc_filename)
    
    save_documentation(output_path, doc_content)
    print(f"   💾 Saved to: {output_path}")
    
    return {
        "current_file_index": current_index + 1,
        "messages": [HumanMessage(content=f"Saved documentation to: {output_path}")]
    }

def should_continue(state: AgentContext) -> str:
    """Decide whether to continue processing files or end."""
    file_paths = state["file_paths"]
    current_index = state["current_file_index"]
    
    if current_index >= len(file_paths):
        return "end"
    return "continue"

# Build the agent graph
agent_builder = StateGraph(AgentContext)

# Add nodes
agent_builder.add_node("extract_metadata", extract_metadata_node)
agent_builder.add_node("find_sources", find_sources_node)
agent_builder.add_node("analyze_file", analyze_file_node)
agent_builder.add_node("categorize_file", categorize_file_node)
agent_builder.add_node("generate_docs", generate_documentation_node)
agent_builder.add_node("save_docs", save_documentation_node)

# Add edges
agent_builder.add_edge(START, "extract_metadata")
agent_builder.add_edge("extract_metadata", "find_sources")
agent_builder.add_edge("find_sources", "analyze_file")
agent_builder.add_edge("analyze_file", "categorize_file")
agent_builder.add_edge("categorize_file", "generate_docs")
agent_builder.add_edge("generate_docs", "save_docs")

# Add conditional edge to loop through files
agent_builder.add_conditional_edges(
    "save_docs",
    should_continue,
    {
        "continue": "analyze_file",
        "end": END
    }
)

agent_graph = agent_builder.compile()

def generate_documentation(
    project_path: str = ".",
    codebase_extensions: list[str] = None,
    output_dir: str = "docs",
    topics: list[str] = None
) -> dict:
    """Run the documentation generation agent.
    
    Args:
        project_path: Path to the project to document
        codebase_extensions: List of file extensions to process (default: [".py"])
        output_dir: Directory to save documentation (default: "docs")
        topics: Optional list of topics to organize documentation by
    
    Returns:
        Final state of the agent after processing all files
    """
    if codebase_extensions is None:
        codebase_extensions = [".py"]
    
    if topics is None:
        topics = []
    
    initial_state = {
        "project_path": project_path,
        "codebase_extensions": codebase_extensions,
        "output_dir": output_dir,
        "topics": topics,
        "file_paths": [],
        "current_file_index": 0,
        "file_symbols": {},
        "file_docstrings": {},
        "file_variables": {},
        "file_functions": {},
        "file_classes": {},
        "documentation": {},
        "file_topics": {},
        "file_relevance": {},
        "doc_filenames": {},
        "package_metadata": {},
        "messages": []
    }
    
    result = agent_graph.invoke(initial_state)
    
    # Print final summary
    print("\n" + "="*60)
    print("✅ Documentation generation complete!")
    doc_count = len([k for k, v in result.get('file_relevance', {}).items() if v])
    print(f"📚 Generated {doc_count} documentation files in: {output_dir}")
    print("="*60 + "\n")
    
    return result