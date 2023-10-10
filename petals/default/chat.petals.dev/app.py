import hivemind
from flask import Flask
from flask_cors import CORS
from flask_sock import Sock
import psutil
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

import config

#MADI

from transformers import AutoTokenizer, GenerationConfig, pipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
import os
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import ChatPromptTemplate

# from dotenv import load_dotenv
from chromadb.config import Settings
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

logger = hivemind.get_logger(__file__)

models = {}
for model_info in config.MODELS:
    logger.info(f"Loading tokenizer for {model_info.repo}")
    tokenizer = AutoTokenizer.from_pretrained(model_info.repo, add_bos_token=False, use_fast=True)

    logger.info(f"Loading model {model_info.repo} with adapter {model_info.adapter} and dtype {config.TORCH_DTYPE}")
    # We set use_fast=False since LlamaTokenizerFast takes a long time to init
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_info.repo,
        active_adapter=model_info.adapter,
        torch_dtype=config.TORCH_DTYPE,
        initial_peers=config.INITIAL_PEERS,
        max_retries=3,
    )
    model = model.to(config.DEVICE)
    generation_config = GenerationConfig.from_pretrained(model_info.repo)
    max_ctx_size = 2048 #2048
    kwargs = {
            "n_ctx": max_ctx_size,
            "max_tokens": max_ctx_size,
            "n_threads": psutil.cpu_count(logical=False),
            "max_tokens": max_ctx_size
    }
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        model_kwargs=kwargs,
        use_fast=True,
        max_new_tokens=30,
        do_sample=False,
        #use_cache=False,
        device=torch.device('cuda') #config.DEVICE #"cuda:0"
        )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": config.DEVICE})
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,

    )
    retriever = db.as_retriever(search_kwargs={'k': 4})
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    qa = RetrievalQA.from_chain_type(
        llm=local_llm, #model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        #chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    model_name = model_info.name
    if model_name is None:  # Use default name based on model/repo repo
        model_name = model_info.adapter if model_info.adapter is not None else model_info.repo
    models[model_name] = model,tokenizer,qa #local_llm,embeddings

logger.info("Starting Flask app")
app = Flask(__name__)
CORS(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)


@app.route("/")
def main_page():
    return app.send_static_file("index.html")


import http_api
import websocket_api
