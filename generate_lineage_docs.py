import os
import re
import yaml
import streamlit as st
from graphviz import Digraph
from utils_openai_summary import get_openai_summary  # ⬅️ You must have this script ready
from pyvis.network import Network
import streamlit.components.v1 as components

# Extract references using regex
def extract_refs(sql_text):
    return re.findall(r"ref\(['\"]([\w_]+)['\"]\)", sql_text)

# Load all model SQL files and capture their references
def load_model_files(base_path):
    model_map = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".sql"):
                model_name = file.replace(".sql", "")
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    refs = extract_refs(content)
                    model_map[model_name] = {
                        "path": os.path.relpath(file_path, base_path),
                        "content": content,
                        "refs": refs,
                        "description": "",
                        "columns": []
                    }
    return model_map

# Load metadata from YAML files
def load_yml_metadata(base_path, model_map):
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".yml"):
                yml_path = os.path.join(root, file)
                with open(yml_path, 'r') as f:
                    try:
                        data = yaml.safe_load(f)

                        if not isinstance(data, dict):
                            print(f"⚠️ Skipped (not a dict): {file} - Loaded as {type(data).__name__}")
                            continue

                        models_section = data.get('models')
                        if not models_section:
                            print(f"ℹ️ Skipped (no 'models'): {file}")
                            continue

                        for model in models_section:
                            name = model.get('name')
                            if name in model_map:
                                model_map[name]['description'] = model.get('description', '')
                                model_map[name]['columns'] = model.get('columns', [])
                    except Exception as e:
                        print(f"❌ YAML Load Error in {file}: {e}")


# 🧠 DAG Drawing: Color-coded based on dependency direction
def draw_dag(model_map, selected_model):
    dot = Digraph()
    visited_upstream = set()
    visited_downstream = set()

    def add_upstream(model_name):
        if model_name in visited_upstream or model_name not in model_map:
            return
        visited_upstream.add(model_name)

        # Selected is green
        color = 'green' if model_name == selected_model else 'lightblue'
        dot.node(model_name, style='filled', fillcolor=color)

        for ref in model_map[model_name]["refs"]:
            if ref not in model_map:
                dot.node(ref, style='filled', fillcolor='red')
            else:
                dot.node(ref, style='filled', fillcolor='lightblue')
                dot.edge(ref, model_name)
            add_upstream(ref)

    def add_downstream(current):
        if current in visited_downstream:
            return
        visited_downstream.add(current)

        for model, details in model_map.items():
            if current in details["refs"]:
                dot.node(model, style='filled', fillcolor='orange')
                dot.edge(current, model)
                add_downstream(model)

    add_upstream(selected_model)
    add_downstream(selected_model)

    return dot

# 🚀 Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("🧬 DBT Model Lineage Visualizer + AI Summary")

    dbt_project_path = st.text_input("📁 Enter path to your DBT models folder", value="/Users/parthgupta/Documents/kortex-dbt-kna")

    if not os.path.exists(dbt_project_path):
        st.warning("Path not found.")
        return

    model_map = load_model_files(dbt_project_path)
    load_yml_metadata(dbt_project_path, model_map)

    if not model_map:
        st.warning("No models found.")
        return

    selected_model = st.selectbox("🎯 Pick your top-level model to visualize", sorted(model_map.keys()))

    # 🧬 Graph Section
    st.subheader("📊 Model Lineage DAG (Upstream / Downstream)")
    dag = draw_dag(model_map, selected_model)
    st.graphviz_chart(dag)

    # ✍️ Summary Section
    st.subheader("📄 AI-Powered Model Summaries")
    def render_summary(model_name, level=0, visited=None):
        if visited is None:
            visited = set()
        if model_name in visited:
            return
        visited.add(model_name)

        model_info = model_map.get(model_name)
        if not model_info:
            return

        st.markdown(f"{'—' * level} 📄 **Model:** `{model_name}`")
        st.markdown(f"{' ' * level * 2}📍Path: `{model_info['path']}`")

        summary = get_openai_summary(model_info["content"])
        st.markdown(f"{' ' * level * 2}📝 **Summary:** {summary}")

        for ref in model_info["refs"]:
            render_summary(ref, level + 1, visited)

    render_summary(selected_model)

if __name__ == "__main__":
    main()
