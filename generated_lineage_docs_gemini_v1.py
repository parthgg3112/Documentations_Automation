import os
import re
import yaml
import streamlit as st
from graphviz import Digraph
import google.generativeai as genai
import time
from datetime import datetime # To add timestamp to filenames

# --- Configuration ---
# Load API Key securely (Keep your existing logic here)
try:
    GEMINI_API_KEY = ""
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("🛑 Gemini API Key not found. Please set it in Streamlit secrets (GEMINI_API_KEY) or as an environment variable.")
        st.stop()
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        # Initialize the Gemini Model
        # Use a default model, consider making this configurable if needed
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro'
except Exception as e:
    st.error(f"🛑 Error configuring Gemini: {e}")
    st.stop()


# --- Helper Functions ---

# 🧠 Gemini Summary Function (Cached)
@st.cache_data(show_spinner="✨ Generating summary with Gemini...")
def get_gemini_summary(sql_content: str, model_name: str) -> str:
    """Generates a summary for the given SQL content using Gemini."""
    # Keep your existing prompt and Gemini call logic here...
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
        # Add generation config for safety if needed
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]
        # response = gemini_model.generate_content(prompt, safety_settings=safety_settings)

        response = gemini_model.generate_content(prompt)

        if response.parts:
             return response.text
        elif response.prompt_feedback.block_reason:
             # Extract the reason if available
             block_reason_details = getattr(response.prompt_feedback, 'block_reason_message', 'No details provided.')
             return f"⚠️ Summary generation blocked: {response.prompt_feedback.block_reason}. Details: {block_reason_details}"
        else:
             # Check for finish_reason if no parts and no block reason
             finish_reason = getattr(response, 'finish_reason', None)
             if finish_reason and finish_reason != 1: # 1 typically means STOP, others might indicate issues
                 return f"⚠️ Summary generation finished unexpectedly: {finish_reason}"
             return "⚠️ No summary content generated (Unknown reason)."

    except Exception as e:
        # Log the error for debugging if running in a server environment
        # logger.error(f"Gemini API error for model {model_name}: {e}")
        return f"❌ Error generating summary: {e}"

# --- File Loading & Parsing Functions ---
# Keep your existing extract_refs, load_model_files, load_yml_metadata functions here...
# (Ensure they handle encoding='utf-8' and potential errors gracefully)

def extract_refs(sql_text):
    """Finds all `ref('model_name')` patterns in SQL."""
    # Handle variations like double/single quotes, hyphens, dots in model names
    return re.findall(r"ref\(['\"]([\w_.-]+)['\"]\)", sql_text)

def load_model_files(base_path):
    """Scans the directory for .sql files and extracts content/refs."""
    model_map = {}
    try:
        # Use scandir for potentially better performance on large directories
        for entry in os.scandir(base_path):
            if entry.is_dir():
                # Recurse into subdirectories
                model_map.update(load_model_files(entry.path))
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
    except FileNotFoundError:
        st.error(f"Error: Directory not found during model loading: {base_path}")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model files from {base_path}: {e}")
    return model_map


def load_yml_metadata(base_path, model_map):
    """Loads descriptions and column info from schema.yml files."""
    # Iterate through potential schema file names or patterns
    schema_patterns = ["schema.yml", "*.schema.yml", "models.yml", "*.models.yml", "properties.yml", "*.properties.yml"]
    found_files = []
    for pattern in schema_patterns:
        # Use glob to find files matching the pattern recursively if needed, or just os.walk
        for root, _, files in os.walk(base_path):
            for file in files:
                 # Check if file matches common patterns
                 if file.endswith((".yml", ".yaml")) and ("schema" in file or "model" in file or "properties" in file):
                    yml_path = os.path.join(root, file)
                    if yml_path in found_files: continue # Avoid processing same file twice
                    found_files.append(yml_path)

                    try:
                        with open(yml_path, 'r', encoding='utf-8') as f: # Specify encoding
                            data = yaml.safe_load(f)

                        if not isinstance(data, dict):
                            # Log instead of print for cleaner output if deployed
                            # logger.info(f"Skipped YAML (not a dict): {yml_path}")
                            continue

                        models_section = data.get('models')
                        if isinstance(models_section, list):
                            for model in models_section:
                                if isinstance(model, dict):
                                    name = model.get('name')
                                    if name in model_map:
                                        model_map[name]['description'] = model.get('description', '')
                                        # Ensure columns are stored as a list of dicts
                                        columns_data = model.get('columns', [])
                                        model_map[name]['columns'] = columns_data if isinstance(columns_data, list) else []

                    except yaml.YAMLError as e:
                        st.warning(f"⚠️ YAML Parse Error in {yml_path}: {e}")
                    except Exception as e:
                        st.warning(f"⚠️ Error processing YAML {yml_path}: {e}")


# --- DAG Drawing Function ---
# Keep your existing draw_dag function here...
def draw_dag(model_map, selected_model):
    """Generates a Graphviz DAG visualization."""
    dot = Digraph(comment=f'DBT Lineage for {selected_model}')
    dot.attr(rankdir='LR') # Left-to-Right layout
    dot.attr(bgcolor='transparent') # Optional: transparent background

    visited_upstream = set()
    visited_downstream = set()
    nodes_in_graph = set() # Keep track of nodes added

    # Function to add upstream nodes (dependencies of the selected model)
    def add_upstream(model_name):
        nonlocal nodes_in_graph # Allow modification of outer scope variable
        if model_name in visited_upstream:
            return
        if model_name not in model_map:
             # Handle case where a ref points to a model not found in the map
             if model_name not in nodes_in_graph:
                 dot.node(model_name, style='filled', fillcolor='lightcoral', shape='box', label=f"{model_name}\n(Not Found)")
                 nodes_in_graph.add(model_name)
             return # Stop recursion for this path

        visited_upstream.add(model_name)

        color = 'palegreen' if model_name == selected_model else 'lightblue'
        if model_name not in nodes_in_graph:
             dot.node(model_name, style='filled', fillcolor=color, shape='box')
             nodes_in_graph.add(model_name)

        for ref in model_map[model_name].get("refs", []):
            is_ref_found = ref in model_map
            # Determine color for the referenced node *before* recursion
            ref_color = 'lightblue' if is_ref_found else 'lightcoral'
            ref_label = ref if is_ref_found else f"{ref}\n(Not Found)"

            if ref not in nodes_in_graph:
                dot.node(ref, style='filled', fillcolor=ref_color, shape='box', label=ref_label)
                nodes_in_graph.add(ref)
            dot.edge(ref, model_name)
            # Only recurse if the reference was found
            if is_ref_found:
                 add_upstream(ref) # Recurse upstream


    # Function to add downstream nodes (models depending on the selected model)
    def add_downstream(current_model_name):
        nonlocal nodes_in_graph # Allow modification of outer scope variable
        if current_model_name in visited_downstream or current_model_name not in model_map :
            return
        visited_downstream.add(current_model_name)

        # Ensure the current node exists (it should have been added by add_upstream if it's the selected node or an upstream node)
        if current_model_name not in nodes_in_graph:
            # This case should ideally not happen if called correctly, but as a safeguard:
            color = 'palegreen' if current_model_name == selected_model else 'moccasin'
            dot.node(current_model_name, style='filled', fillcolor=color, shape='box')
            nodes_in_graph.add(current_model_name)


        # Find models that ref the current_model_name
        for model, details in model_map.items():
            # Check if 'refs' exists and is iterable
            refs_list = details.get("refs", [])
            if isinstance(refs_list, list) and current_model_name in refs_list:
                # This 'model' depends on 'current_model_name', so 'model' is downstream
                downstream_node_name = model
                if downstream_node_name not in nodes_in_graph: # Ensure downstream node exists
                    dot.node(downstream_node_name, style='filled', fillcolor='moccasin', shape='box')
                    nodes_in_graph.add(downstream_node_name)
                # Add the edge from current (upstream) to the dependent (downstream) model
                dot.edge(current_model_name, downstream_node_name)
                # Recurse further downstream from the newly found downstream node
                add_downstream(downstream_node_name)


    # Start the process from the selected model
    if selected_model in model_map:
        add_upstream(selected_model) # Build upstream lineage first
        # Reset visited_downstream before starting downstream traversal from the selected node
        visited_downstream.clear()
        add_downstream(selected_model) # Build downstream lineage starting from selected
    else:
        st.warning(f"Selected model '{selected_model}' not found in parsed models.")
        # Optionally, draw all models if selected is invalid?
        # for model_name in model_map:
        #    dot.node(model_name, shape='box') # Basic node

    return dot

# --- Report Generation Function ---
def generate_markdown_report(selected_model_name, model_map):
    """Generates a Markdown report string for the selected model and its dependencies."""
    report_lines = []
    selected_model_info = model_map.get(selected_model_name)

    if not selected_model_info:
        return f"# Report for {selected_model_name}\n\nModel not found!"

    report_lines.append(f"# DBT Model Report: `{selected_model_name}`")
    report_lines.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Path:** `{selected_model_info.get('path', 'N/A')}`")
    report_lines.append("\n---\n")

    # --- Selected Model Details ---
    report_lines.append(f"## Selected Model: `{selected_model_name}` Details")

    # YML Description
    if selected_model_info.get('description'):
        report_lines.append("### Description (from YAML)")
        report_lines.append(f"{selected_model_info['description']}\n")
    else:
        report_lines.append("### Description (from YAML)\n_Not provided._\n")

    # Columns
    if selected_model_info.get('columns'):
        report_lines.append("### Columns (from YAML)")
        report_lines.append("| Column | Description |")
        report_lines.append("|---|---|")
        for col in selected_model_info['columns']:
             # Ensure col is a dict before accessing keys
             if isinstance(col, dict):
                 col_name = col.get('name', 'N/A')
                 col_desc = col.get('description', '').replace('\n', ' ') # Avoid newlines in table
                 report_lines.append(f"| {col_name} | {col_desc} |")
             else:
                 report_lines.append(f"| _Invalid Column Entry_ | {col} |") # Log unexpected format
        report_lines.append("\n")
    else:
        report_lines.append("### Columns (from YAML)\n_No columns defined in YAML._\n")


    # AI Summary
    report_lines.append("### AI Summary (Gemini)")
    # Note: This relies on the cache or re-generates if needed
    summary = get_gemini_summary(selected_model_info['content'], selected_model_name)
    report_lines.append(f"{summary}\n")

    # Raw SQL
    report_lines.append("### SQL Code")
    report_lines.append("```sql")
    report_lines.append(selected_model_info.get('content', '-- No SQL content found'))
    report_lines.append("```\n")

    report_lines.append("\n---\n")

    # --- Upstream Dependencies ---
    report_lines.append("## ⬆️ Direct Upstream Dependencies (Inputs)")
    upstream_refs = selected_model_info.get("refs", [])
    if upstream_refs:
        for i, ref_name in enumerate(upstream_refs):
            report_lines.append(f"### {i+1}. Upstream Model: `{ref_name}`")
            ref_info = model_map.get(ref_name)
            if ref_info:
                report_lines.append(f"**Path:** `{ref_info.get('path', 'N/A')}`")
                ref_summary = get_gemini_summary(ref_info['content'], ref_name)
                report_lines.append("**AI Summary:**")
                report_lines.append(f"{ref_summary}\n")
                # Optionally add SQL for dependencies too?
                # report_lines.append("**SQL Code:**")
                # report_lines.append("```sql")
                # report_lines.append(ref_info.get('content', '-- No SQL content found'))
                # report_lines.append("```\n")
            else:
                report_lines.append(f"_Details for model `{ref_name}` not found (missing .sql file or parse error?)._\n")
    else:
        report_lines.append("_This model has no direct `ref()` dependencies identified._\n")

    report_lines.append("\n---\n")

    # --- Downstream Dependencies ---
    report_lines.append("## ⬇️ Direct Downstream Dependencies (Outputs To)")
    downstream_models = []
    for model_name, details in model_map.items():
         # Check if 'refs' exists and is iterable before checking membership
        refs_list = details.get("refs", [])
        if isinstance(refs_list, list) and selected_model_name in refs_list:
             downstream_models.append(model_name)

    if downstream_models:
        for i, dep_name in enumerate(downstream_models):
            report_lines.append(f"### {i+1}. Downstream Model: `{dep_name}`")
            dep_info = model_map.get(dep_name) # Should exist if found above
            if dep_info:
                report_lines.append(f"**Path:** `{dep_info.get('path', 'N/A')}`")
                dep_summary = get_gemini_summary(dep_info['content'], dep_name)
                report_lines.append("**AI Summary:**")
                report_lines.append(f"{dep_summary}\n")
                # Optionally add SQL for dependencies too?
                # report_lines.append("**SQL Code:**")
                # report_lines.append("```sql")
                # report_lines.append(dep_info.get('content', '-- No SQL content found'))
                # report_lines.append("```\n")
            # No else needed here as dep_name comes from model_map keys
    else:
        report_lines.append("_This model is not directly referenced by any other loaded models._\n")

    return "\n".join(report_lines)


# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="DBT Doc & Lineage Explorer")
    st.title("🧬 DBT Model Documentation & Lineage Explorer")
    st.write("Visualize model dependencies and get AI-powered summaries using Google Gemini.")

    # --- User Input ---
    if 'dbt_project_path' not in st.session_state:
        # Provide a sensible default - CHANGE THIS
        st.session_state.dbt_project_path = "/path/to/your/dbt_project/models"

    st.session_state.dbt_project_path = st.text_input(
        "📁 Enter path to your DBT **models** folder",
        value=st.session_state.dbt_project_path,
        key="dbt_path_input" # Add a key for stability
    )
    dbt_project_path = st.session_state.dbt_project_path

    if not dbt_project_path or not os.path.isdir(dbt_project_path):
        st.warning("Please enter a valid path to a directory containing your dbt models (.sql files).")
        st.stop()

    # --- Data Loading (Cached) ---
    @st.cache_data(show_spinner="Loading dbt models...")
    def load_all_models(path):
        model_files = load_model_files(path)
        if not model_files: # Early exit if no SQL files found
            st.warning(f"No DBT models (.sql files) found in: {path}")
            return {} # Return empty dict

        # Search for YAMLs in the models dir and potentially one level up (project root)
        project_root = os.path.dirname(path)
        if project_root != path: # Avoid loading same dir twice if path is already root
             load_yml_metadata(project_root, model_files)
        load_yml_metadata(path, model_files)
        return model_files

    try:
        model_map = load_all_models(dbt_project_path)
        # Check again if model_map is empty after loading attempts
        if not model_map:
            # load_all_models might have shown a warning, but stop execution here.
            st.stop()
    except Exception as e:
        st.error(f"An error occurred loading models: {e}")
        st.stop()

    # --- Model Selection ---
    sorted_model_names = sorted(model_map.keys())
    # Use session state for selected model to preserve selection across reruns
    if 'selected_model' not in st.session_state or st.session_state.selected_model not in sorted_model_names:
         st.session_state.selected_model = sorted_model_names[0] if sorted_model_names else None

    # Use the session state value, update it on change
    selected_model = st.selectbox(
        "🎯 Select a model to explore",
        sorted_model_names,
        index=sorted_model_names.index(st.session_state.selected_model) if st.session_state.selected_model in sorted_model_names else 0,
        key='model_selector' # Add a key
    )
    # Update session state if selection changes
    st.session_state.selected_model = selected_model


    if not selected_model or selected_model not in model_map:
        st.error("Please select a valid model.")
        st.stop()

    selected_model_info = model_map[selected_model]

    # --- Display Area ---
    st.markdown("---") # Separator before main content

    # Generate DAG first as it might be needed for download
    try:
        dag = draw_dag(model_map, selected_model)
    except Exception as e:
        st.error(f"Error generating DAG: {e}")
        dag = None # Set dag to None if error occurs

    col1, col2 = st.columns([2, 3]) # Give more space to documentation

    with col1:
        st.subheader("📊 Model Lineage DAG")
        st.write("`Green`: Selected, `Blue`: Upstream, `Orange`: Downstream, `Red`: Not Found")
        if dag:
            try:
                st.graphviz_chart(dag, use_container_width=True)

                # --- Download Button for Graph ---
                try:
                    graph_bytes = dag.pipe(format='png')
                    st.download_button(
                        label="💾 Download Lineage Graph (.png)",
                        data=graph_bytes,
                        file_name=f"dbt_lineage_{selected_model}_{datetime.now().strftime('%Y%m%d')}.png",
                        mime="image/png",
                        key="download_graph"
                    )
                except Exception as render_err:
                     st.warning(f"Could not render graph for download: {render_err}")

            except Exception as graphviz_err:
                 st.error(f"Error displaying DAG: {graphviz_err}")
        else:
            st.warning("Could not generate DAG visualization.")


        # Display Raw SQL in an expander
        with st.expander("View Raw SQL Code for Selected Model"):
            st.code(selected_model_info.get('content', '# No content found'), language='sql')

    with col2:
        st.subheader(f"📄 Documentation for: `{selected_model}`")
        st.markdown(f"**Path:** `{selected_model_info.get('path', 'N/A')}`")

        # --- Download Button for Report ---
        try:
            # Generate report data (this will trigger cached Gemini calls if needed)
            markdown_report = generate_markdown_report(selected_model, model_map)
            st.download_button(
                label="📝 Download Full Report (.md)",
                data=markdown_report.encode('utf-8'), # Encode string to bytes
                file_name=f"dbt_report_{selected_model}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                key="download_report"
            )
        except Exception as report_err:
             st.warning(f"Could not generate report data for download: {report_err}")


        st.markdown("---") # Separator
        tab_titles = ["Overview & AI Summary", "Description & Columns", "Upstream Dependencies", "Downstream Dependencies"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        with tab1:
            st.markdown("#### ✨ AI Summary (Gemini)")
            summary = get_gemini_summary(selected_model_info['content'], selected_model)
            st.markdown(summary)

        with tab2:
             # YML Description
            st.markdown("#### 📝 Model Description (from YAML)")
            if selected_model_info.get('description'):
                st.markdown(selected_model_info['description'])
            else:
                st.info("No description found in YAML.")

             # Columns
            st.markdown("#### 🏛️ Columns (from YAML)")
            if selected_model_info.get('columns'):
                cols_data = []
                for col in selected_model_info['columns']:
                     # Check if col is a dictionary before processing
                     if isinstance(col, dict):
                         cols_data.append({
                             "Column": col.get('name', 'N/A'),
                             "Description": col.get('description', '')
                         })
                     else:
                         # Handle cases where column entry might not be a dictionary
                         cols_data.append({
                             "Column": "Invalid Entry",
                             "Description": str(col) # Display the raw entry
                         })

                if cols_data:
                    st.dataframe(cols_data, use_container_width=True, height=300) # Set height
                else:
                     st.info("No valid column definitions found in YAML.") # Changed message slightly
            else:
                st.info("No columns section found in YAML for this model.")


        # --- Dependencies Tabs ---
        with tab3:
            st.markdown("#### ⬆️ Direct Upstream Models (Inputs)")
            upstream_refs = selected_model_info.get("refs", [])
            if upstream_refs:
                for ref_name in upstream_refs:
                    with st.expander(f"`{ref_name}`"):
                        ref_info = model_map.get(ref_name)
                        if ref_info:
                            st.markdown(f"**Path:** `{ref_info.get('path', 'N/A')}`")
                            ref_summary = get_gemini_summary(ref_info['content'], ref_name)
                            st.markdown("**AI Summary:**")
                            st.markdown(ref_summary)
                            with st.popover("View SQL"): # Use popover for less clutter
                                 st.code(ref_info.get('content', '# No content found'), language='sql')
                        else:
                            st.warning(f"Details for model `{ref_name}` not found.")
            else:
                st.info("This model has no direct `ref()` dependencies identified.")

        with tab4:
             st.markdown("#### ⬇️ Direct Downstream Models (Outputs To)")
             # Recalculate or fetch downstream models
             downstream_models = []
             for model_name, details in model_map.items():
                 refs_list = details.get("refs", [])
                 if isinstance(refs_list, list) and selected_model in refs_list:
                     downstream_models.append(model_name)

             if downstream_models:
                 for dep_name in downstream_models:
                     with st.expander(f"`{dep_name}`"):
                         dep_info = model_map.get(dep_name)
                         if dep_info:
                             st.markdown(f"**Path:** `{dep_info.get('path', 'N/A')}`")
                             dep_summary = get_gemini_summary(dep_info['content'], dep_name)
                             st.markdown("**AI Summary:**")
                             st.markdown(dep_summary)
                             with st.popover("View SQL"): # Use popover for less clutter
                                 st.code(dep_info.get('content', '# No content found'), language='sql')
                         # No else needed here
             else:
                 st.info("This model is not directly referenced by any other loaded models.")


if __name__ == "__main__":
    main()
