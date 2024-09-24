# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen2-VL-7B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def llm(prompt):
#     messages = [
#     {"role": "system", "content": "You are PlantGenic AI. Your task is to give as much information about the input you receive as possible."},
#     {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response
import ollama
def llm(prompt):
    return ollama.chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': prompt,
  },
])