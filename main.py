# import streamlit as st
# from llama_index.llms.ollama import Ollama
# from llama_parse import LlamaParse  ##for extracting data from pdf
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate ##converting data into vector database fro faster access
# from llama_index.core.embeddings import resolve_embed_model ##mapping the data based on what information it is about
# ##creating tools for our agent
# from llama_index.core.tools import QueryEngineTool, ToolMetadata
# ##################
# from llama_index.core.agent import ReActAgent
# from pydantic import BaseModel  ##converts input string into a structured format which in this case is writing a code
# from llama_index.core.output_parsers import PydanticOutputParser
# from llama_index.core.query_pipeline import QueryPipeline  ##run a variety of modules and connect them in sequence
# from prompts import context, code_parser_template
# from code_reader import code_reader
# from dotenv import load_dotenv ##load environment variables
# import ast ##gives python code
# import os


# ##llama parse cannot handle python files
# load_dotenv()

# st.title("Chitter Chatter")
# st.sidebar.title("Options")

# # ##
# # llm=Ollama(model="mistral", request_timeout=1200.0)

# # parser= LlamaParse(result_type="markdown")

# # file_extractor= {".pdf": parser} 
# # documents= SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()


# # embed_model= resolve_embed_model("local:BAAI/bge-m3")  ##download embed model locally
# # vector_index= VectorStoreIndex.from_documents(documents, embed_model=embed_model) ##convert data into vector database
# # query_engine =vector_index.as_query_engine(llm=llm)  ##pass mistral llm model fro querying

# # # result=query_engine.query("what are some of the routes in api?")
# # # print(result)

# # tools = [
# #     QueryEngineTool(
# #     query_engine=query_engine,
# #     metadata=ToolMetadata(
# #         name="api_documentation", ##be specific about name and description
# #         description="this gives documentation about code for an API. Use this for reading docs for the API",
# #         ),
# #     ),
# #     code_reader,
# # ]

# # code_llm= Ollama(model="codellama") ## ollama model that can generate code
# # agent=ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# # class CodeOutput(BaseModel):
# #     code:str
# #     description:str
# #     filename:str

# # parser= PydanticOutputParser(CodeOutput) ## format the input in CodeOutput format
# # json_prompt_str=parser.format(code_parser_template)
# # json_prompt_tmpl= PromptTemplate(json_prompt_str) ##validates input prompt
# # output_pipeline= QueryPipeline(chain=[json_prompt_tmpl, llm])



# # while (prompt := input("Enter a prompt (q to quit): ")) != "q":
# #     retries = 0

# #     while retries < 3:
# #         try:
# #             result = agent.query(prompt)
# #             next_result = output_pipeline.run(response=result)
# #             cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
# #             break
# #         except Exception as e:
# #             retries += 1
# #             print(f"Error occured, retry #{retries}:", e)

# #     if retries >= 3:
# #         print("Unable to process request, try again...")
# #         continue


# #     print("Code generated")
# #     print(cleaned_json["code"])

# #     print("\n\nDescription", cleaned_json["description"])

# #     filename=cleaned_json["filename"]
# #     try:
# #         with open(os.path.join("output", filename), "w") as f:
# #             f.write(cleaned_json["code"])
# #         print("Saved file", filename)
# #     except:
# #         print("Error saving file...")

# # ##


import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
from dotenv import load_dotenv
import ast
import os

# Load environment variables
load_dotenv()

# Initialize the LLM model with a timeout
llm = Ollama(model="mistral", request_timeout=1200.0)

# Set up the document parser for PDF files
parser = LlamaParse(result_type="markdown")

# Define file extractors for different file types
file_extractor = {".pdf": parser}

# Load documents from the "data" directory
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Resolve the embedding model
embed_model = resolve_embed_model("local:BAAI/bge-m3")

# Convert documents into a vector database for faster querying
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Create a query engine using the vector index and Mistral model
query_engine = vector_index.as_query_engine(llm=llm)

# Define tools for the agent
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This gives documentation about code for an API. Use this for reading docs for the API.",
        ),
    ),
    code_reader,
]

# Initialize the code generation LLM model
code_llm = Ollama(model="codellama")

# Create the ReActAgent with the tools
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# Define the output format using Pydantic
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)

# Create the query pipeline for generating output
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

# Streamlit UI
st.title("PDF Chatbot with Mistral Model")
st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Handle file upload and process the document
    out_dir="data"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    st.write("Processing uploaded PDF...")
    bytes_data = uploaded_file.read()
    with open(os.path.join(out_dir, uploaded_file.name), "wb") as f:
        f.write(bytes_data)


    documents = SimpleDirectoryReader([uploaded_file], file_extractor=file_extractor).load_data()
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    query_engine = vector_index.as_query_engine(llm=llm)
    
    tools[0].query_engine = query_engine  # Update the query engine tool with the new engine
    
    prompt = st.chat_input("Enter a prompt (q to quit):")

    if prompt:
        message=st.chat_message("ai")
        retries = 0

        while retries < 3:
            try:
                # Query the agent with the user's input
                result = agent.query(prompt)
                
                # Run the output pipeline and clean up the result
                next_result = output_pipeline.run(response=result)
                cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
                break
            except Exception as e:
                retries += 1
                message.error(f"Error occurred, retry #{retries}: {e}")

        if retries >= 3:
            message.error("Unable to process request, try again...")
        else:
            # Display the generated code and description
            message.markdown("**Code generated**")
            message.code(cleaned_json["code"], language="python")
            message.markdown("**Description**")
            message.write(cleaned_json["description"])

            # Save the generated code to a file
            filename = cleaned_json["filename"]
            try:
                output_dir = "output"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(os.path.join(output_dir, filename), "w") as f:
                    f.write(cleaned_json["code"])
                st.success(f"Saved file: {filename}")
            except:
                st.error("Error saving file...")
