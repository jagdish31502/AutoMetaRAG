import openai
import json
import re
from configparser import ConfigParser
import os
from configparser import ConfigParser
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

def extract_json(text):
    # Regex pattern to find JSON objects
    json_pattern = r'\{[^{}]*\}'

    # Find all matches in the text
    matches = re.findall(json_pattern, text)

    json_objects = []
    for match in matches:
        try:
            json_object = json.loads(match)
            json_objects.append(json_object)
        except json.JSONDecodeError:
            continue

    # Merge the two if exactly two are found
    if len(json_objects) == 2:
        merged = {**json_objects[0], **json_objects[1]}  # latter keys override former
        return [merged, json_objects[1]]

    return json_objects

def generate_metadata(document_info, user_queries, api_key):
    prompt = (''' I am indexing a large dataset for better retrieval. I want to break the data into chunks and index the chunks of data and store metadata of each chunk for better and accurate retrieval.

I will give you information about my dataset and list of probable questions that user may ask. It's not exact questions but just a set of examples. Analyse the data and type of probable questions and give me two lists of metadata that I should extract from each chunk of data for better retrieval.

One list will have metadata that can be extracted from first chunk such as file name, library name, date of file creation etc. Understand that this metadata will remain same for chunks within same file but will change when we index another file.

Another list will have metadata that are specific to that chunk of data like keywords, details or something specific depending upon dataset. But it should be metadata version of entire data chunk.

Your response should only contain two jsons of metadata to be extracted.


For example:

Dataset information: Documentation of python libraries like OpenCV and many others
LIst of probable questions: What is the use of cv2.imshow, How to install opencv-python, What are the keyword arguments for cv2.imread, how to use np.array ?

Analysis: since there are many libraries, one json will have library name, library version and last updated date. Since these data can be extracted
from first chunk of each file. Another json will be specific to the data inside each library documentation so it should have function name and function utility summary.

Your response: {"library_name":"", "library_version":"", "last_updated_date":""}, {"function_name": "", "function_utility_summary":""}

<example over>

Dataset information:''' + document_info +
'''Probable questions: ''' + user_queries )


    main_prompt = [
        {"role": "system", "content": "You are an expert LLM Engineer and Consultant."},
        {"role": "user", "content": prompt}
    ]
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=main_prompt,
        temperature=0
    )

    # Parse the response to JSON metadata format
    metadata_suggestion = response.choices[0].message.content
    return extract_json(metadata_suggestion), metadata_suggestion

def process_documents(documents, json1, json2, openai_api_key):
    """
    Process a list of documents, extracting JSON structures based on the file name similarity and merging them before sending to OpenAI.

    :param documents: List of LlamaIndex Document objects.
    :param json_formats: Tuple of two JSON format strings (json1, json2) that the user wants to extract.
    :param openai_api_key: OpenAI API key for making requests.
    :return: Dictionary of document IDs and their extracted and merged JSONs.
    """
    client = openai.OpenAI(api_key=openai_api_key)
    results = {}
    previous_file_name = None
    last_json1 = None
    # print(f"@@@@JSON Formats: {json_formats}")

    for doc in documents:
        if doc.metadata['file_name'] == previous_file_name:
            # Same file name as the previous document, use the last JSON1 and request only JSON2
            json_to_extract = json2
        else:
            # Different file name, request both JSONs
            json_to_extract = json1
        # print(f"############ Text: {doc.text}")
        # Prepare the OpenAI prompt
        prompt = f"Evaluate the text data and extract required json parameters to create metadata. Understand the overall context of the text and fill up the json metadata strictly as per the structures defined. Respond with Json Only. Nothing extra. Generate one single JSON output, If I give you two json structures, combine them into one JSON Structure: {json_to_extract}\n\nText:\n{doc.text}"
        main_prompt = [
            {"role": "system", "content": "You are an expert JSON extractor and data analyst."},
            {"role": "user", "content": prompt}
        ]
        # print(f"#####Main prompt: {main_prompt}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=main_prompt,
            temperature=0
        )
        # print(f"#####Response: {response}")
        response = response.choices[0].message.content
        raw_response = response.replace('\n', " ")
        # print(f"raw_response: {raw_response}")
        # Extract JSON and attempt to merge
        try:
            # Extract JSON using regex from the response content

            # json_pattern = r'\{.*?\}'
            json_pattern = r'\{[^{}]*\}'
            extracted_json = re.findall(json_pattern, response)[0]
            extracted_dict = json.loads(extracted_json)

            # Merge JSONs as required
            if doc.metadata['file_name'] == previous_file_name and last_json1 is not None:
                merged_json = {**json.loads(last_json1), **extracted_dict}
            else:
                last_json1 = json.dumps(extracted_dict)  # Update last_json1 with new JSON
                merged_json = extracted_dict  # This includes both JSON1 and JSON2

            # print(merged_json)

            results[doc.id_] = json.dumps(merged_json)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for document {doc.id_}: {e}")
        except KeyError as e:
            print(f"Key error: {e} in document {doc.id_}")
        except Exception as e:
            print(f"An error occurred: {e}")

        previous_file_name = doc.metadata['file_name']

    return results

def get_qdrant_collection(client, collection_name):
    # Check if collection exists
    if collection_name in [collection.name for collection in client.get_collections().collections]:
        client.delete_collection(collection_name=collection_name)

    # Create the collection again
    client.create_collection(collection_name=collection_name, vectors_config={"size": 384, "distance": "Cosine"})
    message = "new collection created"
    return message

# Function to extract unique values per key
def extract_unique_nested_values(json_data): 
    unique_values = {}

    # Iterate through each document
    for document in json_data.values():
        # Convert the stringified JSON data into a dictionary
        nested_data = json.loads(document)

        # Process each key-value pair in the nested JSON
        for key, value in nested_data.items():
            if key not in unique_values:
                unique_values[key] = set()
            unique_values[key].add(str(value))

    # Convert sets back to lists for JSON serialization
    return {key: list(values) for key, values in unique_values.items()}

def filter_metadata_by_query(unique_values_json, user_query, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)

    prompt = ''' You will be given a query and master data JSON. The query is to be used for performing hybrid search on a vector database.
    Your job is to analyse the query and respond with key-value pairs that you can find out from the master data JSON. Don't focus on answering
    the query. Just find out which key-value pairs match the master data.

    For example: Master json is {'price' : ['0$-100$', '100$-500$', '500$-1000$', '1000$+'], 'product_category': ['clothes, accessories', 'mobiles, laptops', 'grocery, essentials']}

    Query is "What are some good options for Mens black Tshirt"

    Your response should be {'product_category':'clothes, accessories'}

    Here is the master data json: ''' + json.dumps(unique_values_json) + '''User Query: ''' + user_query + '''Your response: '''

    main_prompt = [
        {"role": "system", "content": "You are an expert JSON extractor and data analyst."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=main_prompt,
        temperature=0
    )

    try:
        json_pattern = r'\{.*?\}'
        extracted_json = re.findall(json_pattern, response.choices[0].message.content, re.DOTALL)[0]
        relevant_entries = json.loads(extracted_json)
        return relevant_entries

    except json.JSONDecodeError:
        return {"error": "❌ Failed to decode response from OpenAI"}



