# icd10_suggester.py

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Read icd10 csv
def load_icd10_data(csv_path="/User/dell/Documents/idc10/icd10_codes.csv"):
    df = pd.read_csv(csv_path)  # expects columns: 'code', 'description'
    icd_dict = dict(zip(df['description'], df['code']))
    return icd_dict, list(icd_dict.keys())

# Ask user query
def build_prompt(user_input):
    return f"""Please enter symptoms:Symptoms/Diagnosis: {user_input}"""

# Load the model mistralai/Mistral-7B-Instruct-v0.1 which has better contentual understanding od medical terms
def load_llm(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

# Process the query
def query_llm(pipe, user_input):
    prompt = build_prompt(user_input)
    output = pipe(prompt, max_new_tokens=128, do_sample=False)
    return output[0]['generated_text']

# Perform the vector embedding match of query
def embedding_match(query, icd_descriptions, icd_dict):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    icd_embeddings = embedder.encode(icd_descriptions, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, icd_embeddings)[0]
    top_idx = int(scores.argmax())
    best_desc = icd_descriptions[top_idx]
    best_code = icd_dict[best_desc]
    return best_code, best_desc

# start the execution
def main():
    icd_dict, icd_descriptions = load_icd10_data("icd10_codes.csv")
    llm_pipe = load_llm()    
  user_input = input("Enter any symptom:")

    llm_response = query_llm(llm_pipe, user_input)
    code, desc = embedding_match(user_input, icd_descriptions, icd_dict)
    print(f"suggested ICD-10 Codes: {code}\nDescription: {desc}")

if __name__ == "__main__":
    main()
