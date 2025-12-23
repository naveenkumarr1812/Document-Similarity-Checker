import os
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
import dotenv

dotenv.load_dotenv()

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat kohli'

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

result = client.sentence_similarity(
    sentence=query,
    other_sentences=documents,
    model="sentence-transformers/all-MiniLM-L6-v2"
)


best_index = result.index(max(result))
best_document = documents[best_index]

print(best_document)



