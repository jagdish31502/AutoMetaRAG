import streamlit as st
import openai
import json
import re
from configparser import ConfigParser
import os
from configparser import ConfigParser
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from utils.functions import extract_json, generate_metadata, process_documents

api_key='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'

# Create necessary folders
os.makedirs("data", exist_ok=True)

# Streamlit UI
# st.title("🔍 Metadata & Data Uploader with LlamaIndex")
st.markdown("Upload a `config.ini` to extract metadata, and upload any `.txt`, `.md`, `.pdf`, etc. files to load into LlamaIndex.")

# --- Upload config.ini ---
st.header("📄 Upload config.ini")
uploaded_file = st.file_uploader("Upload config.ini", type="ini")

if uploaded_file:
    config = ConfigParser()
    config.read_string(uploaded_file.read().decode("utf-8"))

    if config.has_section("Metadata"):
        user_queries = config.get('Metadata', 'probable_questions', fallback="")
        document_info = config.get('Metadata', 'document_info', fallback="")

        with st.spinner("Generating metadata schema..."):
            try:
                metadata_json, raw_response = generate_metadata(document_info, user_queries)
                st.success("Successfully generated metadata schema ")
                st.json(metadata_json)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Missing [Metadata] section in config.ini")

# --- Upload Dataset Files ---
st.header("📂 Upload Dataset Files")

uploaded_files = st.file_uploader("Upload data files (e.g., .txt, .docx, .pdf)", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    # Load files with LlamaIndex
    with st.spinner("Loading documents using LlamaIndex..."):
        try:
            documents = SimpleDirectoryReader("data", filename_as_id=True).load_data()
            st.success(f"✅ documents are uploaded.")

        except Exception as e:
            st.error(f"❌ Error loading documents: {e}")

# --- JSON Format Input ---
st.header("🧩 Define JSON Format for Metadata Extraction")

# default_json1 = '{"document_type": "", "document_title": "", "document_creation_date": "", "department": ""}'
# default_json2 = '{"section_title": "", "section_keywords": "", "section_summary": ""}'

json1_input = st.text_area("Enter JSON Format for File-Level Metadata",  height=70)
json2_input = st.text_area("Enter JSON Format for Chunk-Level Metadata",  height=70)

# When documents are loaded and JSON formats are provided
if json1_input and json2_input and st.button("🚀 Process Documents for Metadata"):
    try:
        documents = SimpleDirectoryReader("data", filename_as_id=True).load_data()
        with st.spinner("Processing documents to extract metadata..."):
            extracted_jsons = process_documents(documents, (json1_input, json2_input), api_key)
        st.success("✅ Metadata extracted successfully!")
        # for doc_id, metadata in extracted_jsons.items():
        #     st.subheader(f"📄 Document ID: {doc_id}")
        #     st.json(json.loads(metadata))

    except Exception as e:
        st.error(f"Error during processing: {e}")


