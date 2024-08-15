from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill" 

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_histrory = [] 

history_string = "\n".join(conversation_histrory) 

input_text = "how many countries are there in the world?" 

inputs = tokenizer(history_string + input_text, return_tensors="pt")

# Generate the response
outputs = model.generate(**inputs)

# Decode the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(response)

 
