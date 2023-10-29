from transformers import BertTokenizer, BertModel


# Load BERT model and tokenizer
model_name = 'bert-base-uncased'  # You can choose a specific variant
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
