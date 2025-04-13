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
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, Range
from utils.functions import generate_metadata, process_documents, get_qdrant_collection, extract_unique_nested_values, filter_metadata_by_query

# Initialize session state
if "extracted_jsons" not in st.session_state:
    st.session_state.extracted_jsons = None

# UI to accept OpenAI API Key
st.subheader("🔐 OpenAI Configuration")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
api_key=openai_api_key

# Create necessary folders
os.makedirs("data", exist_ok=True)

# Streamlit UI
# st.title("🔍 Metadata & Data Uploader with LlamaIndex")
st.markdown("Upload a `config.ini` to extract metadata")

# --- Upload config.ini ---
st.header("📄 Upload config.ini")
uploaded_file = st.file_uploader("Upload config.ini", type="ini")

if uploaded_file and st.button("get metadata schema"):
    config = ConfigParser()
    config.read_string(uploaded_file.read().decode("utf-8"))

    if config.has_section("Metadata"):
        user_queries = config.get('Metadata', 'probable_questions', fallback="")
        document_info = config.get('Metadata', 'document_info', fallback="")

        with st.spinner("Generating metadata schema..."):
            try:
                metadata_json, raw_response = generate_metadata(document_info, user_queries, api_key)
                st.session_state["metadata_schema"] = metadata_json  # ✅ Store in session
                st.success("Successfully generated metadata schema ")
                # st.json(metadata_json)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Missing [Metadata] section in config.ini")

# ✅ Persisted display of metadata schema
if "metadata_schema" in st.session_state:
    st.markdown("### 📦 Metadata Schema")
    st.json(st.session_state["metadata_schema"])

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
        splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=20)
        try:
            entire_docs = SimpleDirectoryReader("data", filename_as_id=True).load_data()
            documents = splitter.get_nodes_from_documents(entire_docs)
            st.success(f"✅ documents are uploaded.")

        except Exception as e:
            st.error(f"❌ Error loading documents: {e}")

# --- JSON Format Input ---
st.header("🧩 Define JSON Format for Metadata Extraction")
json1_input = st.text_area("Enter JSON Format for File-Level Metadata",  height=70)
json2_input = st.text_area("Enter JSON Format for Chunk-Level Metadata",  height=70)
# When documents are loaded and JSON formats are provided
if json1_input and json2_input and st.button("🚀 Process Documents for Metadata"):
    try:
        with st.spinner("Processing documents to extract metadata..."):
            extracted_jsons = process_documents(documents, (json1_input, json2_input), api_key)
            st.session_state.extracted_jsons = extracted_jsons
            with open('data.json', 'w') as f:
                json.dump(extracted_jsons, f)
        st.success("✅ Metadata extracted successfully!")
    except Exception as e:
        st.error(f"Error during processing: {e}")

# Show download button if data is ready
if st.session_state.extracted_jsons:
    st.download_button(
        label="📥 Download extracted metadata",
        data=json.dumps(st.session_state.extracted_jsons, indent=4),
        file_name='data.json',
        mime='application/json',
    )

# Ingest into qdrant
st.subheader("📁 Ingest into database")
QDRANT_URL  = st.text_input("Qdrant URL")
ACCESS_TOKEN  = st.text_input("API Key", type="password")
collection_name = st.text_input("Collection Name", value="AutoRAG")
uploaded_file = st.file_uploader("Upload `data.json`", type=["json"])
# Initialize the Qdrant client
client = QdrantClient(url=QDRANT_URL, api_key=ACCESS_TOKEN)
if uploaded_file and st.button("🚀 Ingest into database"):
    try:
        messgae = get_qdrant_collection(client, collection_name)
        if messgae:
            st.success(messgae)

        extracted_jsons= json.load(uploaded_file)
        st.session_state["extracted_jsons"] = extracted_jsons
        # Initialize the sentence transformer model
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        # Prepare points to be uploaded
        points = []
        index = 0
        unique_id_to_metadata = {
            json.loads(value)["unique_id"]: value for value in extracted_jsons.values()
        }
        for document in documents:
            file_name = document.metadata['file_name']
            if file_name in unique_id_to_metadata:
                metadata = unique_id_to_metadata[file_name]
                vector = encoder.encode(document.text)
                # Create a point with the metadata and the encoded vector
                point = PointStruct(
                    id=index,
                    payload=json.loads(metadata),
                    vector=vector  # Convert numpy array to list
                )
                points.append(point)
            index += 1
        client.upsert(collection_name=collection_name, points=points)
        st.success(f"Successfully ingested {len(points)} documents into the Qdrant collection.")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

#Extract Unique values
st.subheader("🔍 Extract Unique Values from JSON")
# Display results after file is uploaded
if st.button("Extract unique values"):
    try:
        data = st.session_state.get("extracted_jsons", {})
        unique_values_per_key = extract_unique_nested_values(data)
        st.session_state["unique_values_per_key"] = unique_values_per_key
        st.success("✅ Unique values extracted successfully!")
        # st.json(unique_values_per_key)
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📂 Please upload JSON file to extract unique values.")

# ✅ Persisted display of metadata schema
if "unique_values_per_key" in st.session_state:
    st.json(st.session_state["unique_values_per_key"])

# Filter by metadata
st.subheader("🔍 Filter by Unique values")
user_query = st.text_area("💬 Enter your search query", height=70)
# --- Submit Button ---
if st.button("filter by unique values"):
    try:
        unique_values_per_key = st.session_state.get("extracted_jsons", {})
        result = filter_metadata_by_query(unique_values_per_key, user_query, api_key)
        st.success("✅ Filtered metadata based on the query:")
        st.json(result)
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("📌 Please provide query to proceed.")

try:
    # Initialize the sentence transformer model
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    metadata_filter = Filter(
        should=[
            FieldCondition(
                key=list(result.keys())[0],
                match={"value": list(result.values())[0]}
            )
        ]
    )
    st.markdown("### Metadata Filter")
    st.json(metadata_filter)
    st.subheader("🧩 RAG : Pass Metadata Filter + User Query to Qdrant Search")
    query_vector = encoder.encode(user_query).tolist()
    if metadata_filter:
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3,
            query_filter=metadata_filter
        )
        st.success("Search executed successfully!")

        st.subheader("🔍 Top Search Results")
        for hit in hits:
            st.json(hit.payload)
            # st.markdown(f"**ID:** {hit.id}")
            # st.markdown(f"**Score:** {hit.score:.4f}")
            # st.markdown(f"**Title:** {hit.payload.get('section_title', 'N/A')}")
            # st.markdown(f"**Summary:** {hit.payload.get('section_summary', 'N/A')}")
            st.markdown("---")
    # Collect context from retrieved hits
    st.subheader("🤖 RAG - Passing Retrieved Data Chunks to LLM for Final Response")
    context_chunks = []
    for hit in hits:
        context_chunks.append(hit.payload)

    context = "\n\n".join([json.dumps(chunk, indent=2) for chunk in context_chunks])
    # st.markdown("###Context")
    # st.text(context)

    # Construct the prompt
    prompt = f'''Based on the provided context information from the dataset, generate a comprehensive answer for the user query.
    Context: {context}
    User Query: {user_query}'''

    main_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Make the request to OpenAI
        client = openai.OpenAI(api_key=api_key)  # assuming key is collected earlier
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=main_prompt,
            temperature=0
        )

        st.subheader("🧠 Final LLM Response")
        st.markdown(response.choices[0].message.content)

    except Exception as e:
        st.error(f"❌ Error calling OpenAI: {e}")


except Exception as e:
    st.error(f"❌ Error while searching Qdrant: {e}")

