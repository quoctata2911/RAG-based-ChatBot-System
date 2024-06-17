import yaml
import torch
import logging
import argparse
import warnings
import pandas as pd
from tqdm.auto import tqdm
from jsonargparse import CLI
from types import SimpleNamespace
from llama_index.core.schema import TextNode
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Prompt, Settings, VectorStoreIndex
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer


def load_config(config_path='config.yaml'):
    print('-> Loading config file ...')
    cfg = yaml.safe_load(
        open(config_path).read()
    )

    for k,v in cfg.items():
        if type(v) == dict:
            cfg[k] = SimpleNamespace(**v)
    cfg = SimpleNamespace(**cfg)
    return cfg

def get_prompt_template():
    template = (
    "Bạn là trợ lý ảo hữu ích và thông minh được huấn luyên được để trả lời các câu hỏi từ người dùng giữa trên các thông tin ngữ cảnh liên quan được cung cấp\n"
    "Thông tin ngữ cảnh:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Dựa trên những thông tin ngữ cảnh bên trên, hãy trả lời câu hỏi sau: {query_str}\n"
    )
    qa_template = Prompt(template)
    return qa_template

def reset_settings(cfg):
    embed_model =HuggingFaceEmbeddings(
        model_name=cfg.architecture.embedding_model
    )
    Settings.embed_model = embed_model
    Settings.llm = None  

def get_retriever(cfg, prompt_template):
    chunks = pd.read_pickle('processed_chunks.pickle')['chunk'].values.tolist()
    nodes = [TextNode(text=chunk) for chunk in chunks]
    index = VectorStoreIndex(nodes=nodes)
    retriever = index.as_query_engine(
        similarity_top_k=cfg.retrieve.top_k,
        text_qa_template=prompt_template
    )
    return retriever

def load_tokenizer(cfg):
    tokenizer =  AutoTokenizer.from_pretrained(
        cfg.architecture.llm_model,
        token=cfg.architecture.hf_token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_llm(cfg):
    if cfg.architecture.llm_quantized:
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
    else:
        bnb_config = None
            

    llm = AutoModelForCausalLM.from_pretrained(
        cfg.architecture.llm_model,
        torch_dtype=torch.bfloat16,
        device_map=cfg.environment.device,
        token=cfg.architecture.hf_token,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

    return llm.eval()


def vistral_chat(cfg, retriever, tokenizer, language_model):
    while True:
        user_query = input('👨‍🦰 ')
        prompt = retriever.query(user_query).response
        prompt = tokenizer.bos_token + '[INST] ' + prompt + ' [/INST]'
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        input_ids = tokenizer([prompt], return_tensors='pt').to(cfg.environment.device)

        _ = language_model.generate(
            **input_ids,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=cfg.generation.max_new_tokens,
            do_sample=cfg.generation.do_sample,
            temperature=cfg.generation.temperature
        )

        print(20*'---')


def main(config_path):
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Log the start of the process
        logger.info("Starting the process with config file: %s", config_path)
        
        # Load configuration from the file
        config = load_config(config_path)
        
        # Load necessary components
        prompt_template = get_prompt_template()
        
        # Replace OpenAI embed model and llm with custom ones
        reset_settings(config)
        
        # Get retriever
        retriever = get_retriever(config, prompt_template)
        
        # Load tokenizer and language model
        tokenizer = load_tokenizer(config)
        language_model = get_llm(config)
        
        # Start the command line interface
        vistral_chat(config, retriever, tokenizer, language_model)
        
        # Log successful completion
        logger.info("Process completed successfully.")
        
    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", e)
    except Exception as e:
        logger.exception("An error occurred: %s", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some configurations.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)