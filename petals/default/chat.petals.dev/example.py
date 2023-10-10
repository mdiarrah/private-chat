import time
import psutil
from transformers import AutoTokenizer, GenerationConfig, pipeline
from petals import AutoDistributedModelForCausalLM

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings

# from dotenv import load_dotenv
from chromadb.config import Settings
import torch
from transformers import BloomTokenizerFast, get_scheduler

from petals import DistributedBloomForCausalLM

INITIAL_PEERS = [
    "/ip4/57.128.41.243/tcp/31337/p2p/QmbuJziFpMsP7zb7XzHV4jdyrHuMBB4syYuq5KnH6aupXw"
]


ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)
# Choose any model available at https://health.petals.dev
#model_name = "petals-team/StableBeluga2"  # This one is fine-tuned Llama 2 (70B)
#model_name = "meta-llama/Llama-2-70b-chat-hf" #"huggyllama/llama-65b"
#model_name = "bigscience/bloom-7b1-petals"
model_name = "huggyllama/llama-7b"
TUNING_MODE = 'ptune'
NUM_PREFIX_TOKENS = 16
DEVICE = 'cpu'
BATCH_SIZE = 8
LR = 1e-2
WEIGHT_DECAY = 0.0
NUM_SAMPLES = 1000
SEED = 42
MODEL_MAX_LENGTH = 256
'''
tokenizer = BloomTokenizerFast.from_pretrained(model_name)
tokenizer.padding_side = 'right'
tokenizer.model_max_length = MODEL_MAX_LENGTH
model = DistributedBloomForCausalLM.from_pretrained(
    MODEL_NAME,
    pre_seq_len=NUM_PREFIX_TOKENS, 
    tuning_mode=TUNING_MODE
).to(DEVICE)
'''
#tokenizer = BloomTokenizerFast.from_pretrained(model_name)
#tokenizer.padding_side = 'right'
#tokenizer.model_max_length = MODEL_MAX_LENGTH
#logging.info("Local LLM Loaded")

# Connect to a distributed network hosting model layers
#active_adapter="timdettmers/guanaco-7b",
model = AutoDistributedModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32,
         initial_peers=INITIAL_PEERS,max_retries=3,).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name,add_bos_token=False, use_fast=True)#,add_bos_token=False, use_fast=False)
#model = DistributedBloomForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32,pre_seq_len=NUM_PREFIX_TOKENS, 
#tuning_mode=TUNING_MODE).to(DEVICE)
generation_config = GenerationConfig.from_pretrained(model_name)
max_ctx_size = 500 #2048
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
        #max_length=1500,
        #temperature=0,
        #top_p=0.95,
        #repetition_penalty=1.15,
        generation_config=generation_config,
        model_kwargs=kwargs,
        use_fast=True,
        max_new_tokens=40,
        do_sample=False,
    )

local_llm = HuggingFacePipeline(pipeline=pipe)

device_type="cpu"

embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
# Define the Chroma settings
'''
CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)
'''

db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,

   )

#target_source_chunks = 4
#MODEL_N_CTX=1000
retriever = db.as_retriever()
    

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
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    
    # Get the answer from the chain
query = "who is the founder of Hive?"
start = time.time()
res = qa(query)
answer, docs = res["result"], [] #res["source_documents"]
#answer = res["result"]
end = time.time()

print("\n\n> Question:")
print(query)
print(f"\n> Answer (took {round(end - start, 2)} s.):")
print("\n> Answer:")
print(answer)
#print ("\n> Sources:")
#print(docs)

# Run the model as if it were on your computer
'''
inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))
'''
