from prompt_templates import llama3_prompt_msg

from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores import chroma
import torch
from transformers import AutoTokenizer, pipeline,AutoModelForCausalLM, BitsAndBytesConfig
import chromadb
import yaml

with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

def create_llm(
        model_path:str = config["model_path"],
        hf_token:str = config["hf_token"],
        max_tokens:int = config["max_tokens"],
        ):

        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
        )
        #quantized model
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                token = hf_token
        )

        #tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token = hf_token
        )

        text_pipeline = pipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                do_sample = True,
                temperature = 0.2,
                repetition_penalty=1.1,
                return_full_text = False,
                max_new_tokens = max_tokens
        )

        llama_llm = HuggingFacePipeline(pipeline=text_pipeline)



        return llama_llm

def create_embeddings(embedding_path = config["embeddings_path"]):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_path,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    return embedding_model
     
def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(
        memory_key="history",
        chat_memory=chat_history,
        k=3
        )

def create_prompt_from_template(template):
    prompt = PromptTemplate(
        input_variables=["history","question"],
        template=template
    )

    return prompt

def create_llm_chain(llm,chat_prompt):
    return LLMChain(llm=llm,prompt=chat_prompt)

def load_normal_chain():
    return chatChain()

class chatChain:
    def __init__(self):
        llm = create_llm()
        chat_prompt = create_prompt_from_template(llama3_prompt_msg)
        self.llm_chain = create_llm_chain(llm,chat_prompt)

    def run(self,user_input,chat_history):
        return self.llm_chain.invoke(input={"history":"Hi ",
                                            "question":user_input})