from transformers import LukeForEntitySpanClassification, LukeTokenizer

# Define the model name and save paths
model_name = "studio-ousia/luke-base"
model_save_path = "/model"
tokenizer_save_path = "/model/tokenizer"

# Download the model and tokenizer
model = LukeForEntitySpanClassification.from_pretrained(model_name)
tokenizer = LukeTokenizer.from_pretrained(model_name)

# Save the model and tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print("Model and tokenizer downloaded and saved successfully.")
