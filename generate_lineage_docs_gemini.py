import os
import re
import yaml
import streamlit as st
from graphviz import Digraph
import google.generativeai as genai # Import Gemini library
import time # For potential retries or delays

# --- Configuration ---
# Load API Key securely
try:
    # Attempt to load from Streamlit secrets first
    GEMINI_API_KEY = "AIzaSyCvgVKUBF5HaTWVBiDKnZLxkDMmeZcUqGw"
    if not GEMINI_API_KEY:
        # Fallback to environment variable if not in secrets
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.error("🛑 Gemini API Key not found. Please set it in Streamlit secrets (GEMINI_API_KEY) or as an environment variable.")
        st.stop() # Halt execution if no key
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        # Initialize the Gemini Model (use appropriate model name)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro' etc.
except Exception as e:
    st.error(f"🛑 Error configuring Gemini: {e}")
    st.stop()


# --- Helper Functions ---

# 🧠 Gemini Summary Function (Replaces OpenAI)
@st.cache_data(show_spinner="✨ Generating summary with Gemini...") # Cache results
def get_gemini_summary(sql_content: str, model_name: str) -> str:
    """Generates a summary for the given SQL content using Gemini."""
    prompt = f"""
    Analyze the following dbt model SQL code for the model named '{model_name}'.

    Provide a concise summary covering:
    1.  **Purpose:** What is the main goal or objective of this model? What business entity or concept does it represent?
    2.  **Key Logic/Transformations:** Briefly describe the major steps, joins, calculations, or transformations happening in the SQL.
    3.  **Inputs:** Mention the primary source models or tables (based on `ref` or `source` calls if visible, otherwise infer from CTEs/joins).
    4.  **Output:** Describe the final output or structure of the model.

    SQL Code:
    ```sql
    {sql_content}
    ```

    Summary:
    """
    try:
        response = gemini_model.generate_content(prompt)
        # Handle potential safety blocks or empty responses
        if response.parts:
             return response.text
        elif response.prompt_feedback.block_reason:
             return f"⚠️ Summary generation blocked: {response.prompt_feedback.block_reason}"
        else:
             return "⚠️ No summary content generated."

    except Exception as e:
        # Basic retry logic (optional)
        # time.sleep(2)
        # try:
        #     response = gemini_model.generate_content(prompt)
        #     return response.text
        # except Exception as e_retry:
        #      return f"❌ Error generating summary (after retry): {e_retry}"
        return f"❌ Error generating summary: {e}"

# Extract references using regex
def extract_refs(sql_text):
    """Finds all `ref('model_name')` patterns in SQL."""
    return re.findall(r"ref\(['\"]([\w_.-]+)['\"]\)", sql_text) # Handle potential hyphens/dots

# Load all model SQL files and capture their references
def load_model_files(base_path):
    """Scans the directory for .sql files and extracts content/refs."""
    model_map = {}
    # Use scandir for potentially better performance on large directories
    for entry in os.scandir(base_path):
        if entry.is_dir():
            model_map.update(load_model_files(entry.path)) # Recurse into subdirectories
        elif entry.is_file() and entry.name.endswith(".sql"):
            model_name = entry.name.replace(".sql", "")
            file_path = entry.path
            try:
                with open(file_path, "r", encoding='utf-8') as f: # Specify encoding
                    content = f.read()
                    refs = extract_refs(content)
                    model_map[model_name] = {
                        "path": os.path.relpath(file_path, base_path),
                        "content": content,
                        "refs": refs,
                        "description": "", # Initialize description
                        "columns": []      # Initialize columns
                    }
            except Exception as e:
                st.warning(f"⚠️ Error reading file {file_path}: {e}")
    return model_map

# Load metadata from YAML files
def load_yml_metadata(base_path, model_map):
    """Loads descriptions and column info from schema.yml files."""
    for root, _, files in os.walk(base_path):
        for file in files:
            # Common naming conventions for dbt schema files
            if file.endswith((".yml", ".yaml")) and ("schema" in file or "model" in file or "properties" in file):
                yml_path = os.path.join(root, file)
                try:
                    with open(yml_path, 'r', encoding='utf-8') as f: # Specify encoding
                        data = yaml.safe_load(f)

                        if not isinstance(data, dict):
                            # print(f"⚠️ Skipped YAML (not a dict): {file} - Loaded as {type(data).__name__}")
                            continue

                        # Handle different potential structures (e.g., models at top level or nested)
                        models_section = data.get('models')
                        if isinstance(models_section, list):
                            for model in models_section:
                                if isinstance(model, dict):
                                    name = model.get('name')
                                    if name in model_map:
                                        model_map[name]['description'] = model.get('description', '')
                                        model_map[name]['columns'] = model.get('columns', [])
                        # Add checks for other potential structures if needed
                        # elif isinstance(data.get('sources'), list): ... # Handle sources if necessary
                        # elif isinstance(data.get('seeds'), list): ... # Handle seeds if necessary

                except yaml.YAMLError as e:
                    st.warning(f"⚠️ YAML Parse Error in {yml_path}: {e}")
                except Exception as e:
                    st.warning(f"⚠️ Error processing YAML {yml_path}: {e}")

# 🧬 DAG Drawing: Color-coded based on dependency direction
def draw_dag(model_map, selected_model):
    """Generates a Graphviz DAG visualization."""
    dot = Digraph(comment=f'DBT Lineage for {selected_model}')
    dot.attr(rankdir='LR') # Left-to-Right layout

    visited_upstream = set()
    visited_downstream = set()
    nodes_in_graph = set() # Keep track of nodes added

    # Function to add upstream nodes (dependencies of the selected model)
    def add_upstream(model_name):
        if model_name in visited_upstream or model_name not in model_map:
            return
        visited_upstream.add(model_name)
        nodes_in_graph.add(model_name)

        color = 'palegreen' if model_name == selected_model else 'lightblue'
        dot.node(model_name, style='filled', fillcolor=color, shape='box')

        for ref in model_map[model_name].get("refs", []):
            if ref not in model_map:
                # Mark missing refs clearly
                dot.node(ref, style='filled', fillcolor='lightcoral', shape='box')
                nodes_in_graph.add(ref)
            else:
                 # Ensure the referenced node exists before adding edge
                if ref not in nodes_in_graph:
                     dot.node(ref, style='filled', fillcolor='lightblue', shape='box')
                     nodes_in_graph.add(ref)
                dot.edge(ref, model_name)
                add_upstream(ref) # Recurse upstream

    # Function to add downstream nodes (models depending on the selected model)
    def add_downstream(current_model_name):
        if current_model_name in visited_downstream:
            return
        visited_downstream.add(current_model_name)

        # Ensure the current node exists (might be the selected node)
        if current_model_name not in nodes_in_graph:
            color = 'palegreen' if current_model_name == selected_model else 'moccasin' # Use orange/moccasin for downstream
            dot.node(current_model_name, style='filled', fillcolor=color, shape='box')
            nodes_in_graph.add(current_model_name)

        # Find models that ref the current_model_name
        for model, details in model_map.items():
            if current_model_name in details.get("refs", []):
                if model not in nodes_in_graph: # Ensure downstream node exists
                    dot.node(model, style='filled', fillcolor='moccasin', shape='box')
                    nodes_in_graph.add(model)
                dot.edge(current_model_name, model)
                add_downstream(model) # Recurse downstream

    # Start the process from the selected model
    if selected_model in model_map:
        add_upstream(selected_model)
        add_downstream(selected_model) # Start downstream check from selected node
    else:
        st.warning(f"Selected model '{selected_model}' not found in parsed models.")
        # Optionally, draw all models if selected is invalid?
        # for model_name in model_map:
        #    dot.node(model_name, shape='box') # Basic node

    return dot

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="DBT Doc & Lineage Explorer")
    st.title("🧬 DBT Model Documentation & Lineage Explorer")
    st.write("Visualize model dependencies and get AI-powered summaries using Google Gemini.")

    # --- User Input ---
    # Use session state to remember the path
    if 'dbt_project_path' not in st.session_state:
        # Provide a sensible default or leave empty
        st.session_state.dbt_project_path = "/Users/parthgupta/Documents/kortex-dbt-kna/models" #<---- CHANGE THIS DEFAULT

    st.session_state.dbt_project_path = st.text_input(
        "📁 Enter path to your DBT **models** folder",
        value=st.session_state.dbt_project_path
    )
    dbt_project_path = st.session_state.dbt_project_path

    if not dbt_project_path or not os.path.isdir(dbt_project_path):
        st.warning("Please enter a valid path to a directory containing your dbt models (.sql files).")
        st.stop() # Stop if path is invalid

    # --- Data Loading ---
    # Use caching for loading models to speed up reruns if path doesn't change
    @st.cache_data(show_spinner="Loading dbt models...")
    def load_all_models(path):
        model_files = load_model_files(path)
        # Assume YAML files might be anywhere within or above the models dir
        # Go up one level to potentially find project-level schema files
        project_root = os.path.dirname(path)
        load_yml_metadata(project_root, model_files) # Check project root
        load_yml_metadata(path, model_files) # Check models dir itself
        return model_files

    try:
        model_map = load_all_models(dbt_project_path)
    except Exception as e:
        st.error(f"An error occurred loading models: {e}")
        st.stop()

    if not model_map:
        st.warning(f"No DBT models (.sql files) found in the specified path: {dbt_project_path}")
        st.stop()

    # --- Model Selection ---
    sorted_model_names = sorted(model_map.keys())
    selected_model = st.selectbox(
        "🎯 Select a model to explore",
        sorted_model_names,
        index=sorted_model_names.index(sorted_model_names[0]) if sorted_model_names else 0 # Default to first if available
    )

    if not selected_model or selected_model not in model_map:
        st.error("Please select a valid model.")
        st.stop()

    # --- Display Area ---
    selected_model_info = model_map[selected_model]

    col1, col2 = st.columns([1, 1]) # Adjust ratio as needed

    with col1:
        st.subheader("📊 Model Lineage DAG")
        st.write("`Green`: Selected, `Blue`: Upstream (Dependencies), `Orange`: Downstream (Dependents), `Red`: Missing Ref")
        try:
            dag = draw_dag(model_map, selected_model)
            st.graphviz_chart(dag)
        except Exception as e:
            st.error(f"Error generating DAG: {e}")

        # Display Raw SQL in an expander
        with st.expander("View Raw SQL Code"):
            st.code(selected_model_info.get('content', '# No content found'), language='sql')

    with col2:
        st.subheader(f"📄 Documentation for: `{selected_model}`")
        st.markdown(f"**Path:** `{selected_model_info.get('path', 'N/A')}`")

        # --- Selected Model Summary ---
        st.markdown("---")
        st.markdown("#### ✨ Gemini AI Summary")
        summary = get_gemini_summary(selected_model_info['content'], selected_model)
        st.markdown(summary)

        # --- YML Description ---
        if selected_model_info.get('description'):
            st.markdown("---")
            st.markdown("#### 📝 Model Description (from YAML)")
            st.markdown(selected_model_info['description'])

        # --- Columns ---
        if selected_model_info.get('columns'):
            st.markdown("---")
            st.markdown("#### 🏛️ Columns (from YAML)")
            cols_data = []
            for col in selected_model_info['columns']:
                 cols_data.append({
                     "Column": col.get('name', 'N/A'),
                     "Description": col.get('description', '')
                 })
            if cols_data:
                st.dataframe(cols_data, use_container_width=True)
            else:
                st.markdown("_No column descriptions found in YAML._")


        # --- Dependencies ---
        st.markdown("---")
        st.subheader("🔗 Dependencies")

        # Upstream (Models this one depends on)
        st.markdown("#### ⬆️ Direct Upstream Models (Inputs)")
        upstream_refs = selected_model_info.get("refs", [])
        if upstream_refs:
            for ref_name in upstream_refs:
                with st.expander(f"`{ref_name}`"):
                    ref_info = model_map.get(ref_name)
                    if ref_info:
                        st.markdown(f"**Path:** `{ref_info.get('path', 'N/A')}`")
                        ref_summary = get_gemini_summary(ref_info['content'], ref_name)
                        st.markdown("**Summary:**")
                        st.markdown(ref_summary)
                    else:
                        st.warning(f"Details for model `{ref_name}` not found (missing .sql file or parse error?).")
        else:
            st.markdown("_This model has no direct `ref()` dependencies identified._")

        # Downstream (Models that depend on this one)
        st.markdown("#### ⬇️ Direct Downstream Models (Outputs To)")
        downstream_models = []
        for model_name, details in model_map.items():
            if selected_model in details.get("refs", []):
                downstream_models.append(model_name)

        if downstream_models:
            for dep_name in downstream_models:
                 with st.expander(f"`{dep_name}`"):
                    dep_info = model_map.get(dep_name) # Should always exist if found above
                    if dep_info:
                        st.markdown(f"**Path:** `{dep_info.get('path', 'N/A')}`")
                        dep_summary = get_gemini_summary(dep_info['content'], dep_name)
                        st.markdown("**Summary:**")
                        st.markdown(dep_summary)
                    # No else needed here as dep_name comes from model_map keys
        else:
            st.markdown("_This model is not directly referenced by any other loaded models._")


if __name__ == "__main__":
    main()