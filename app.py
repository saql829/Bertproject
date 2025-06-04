import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load data and model
df = pd.read_csv("emails_with_clean_text.csv")
embeddings = np.load("email_embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cleaning function
def final_clean(text):
    try:
        lines = text.split('\n')
        content_lines = []
        for line in lines:
            line = line.strip()
            if re.match(r"^(from:|to:|date:|subject:|message-id:)", line.lower()):
                continue
            if line.lower().startswith(('--', 'thanks', 'regards', 'best')):
                break
            content_lines.append(line)
        content = ' '.join(content_lines)
        content = re.sub(r'\s+', ' ', content).strip().lower()
        return content
    except:
        return ""

# Similarity function
def get_reply_suggestions(new_email_text, top_k=3):
    new_clean = final_clean(new_email_text)
    if not new_clean:
        return ["No valid input"]
    new_embedding = model.encode([new_clean])
    similarities = cosine_similarity(new_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]['clean_body'].tolist()

# Streamlit UI
st.title("📧 Smart Email Reply Suggestions")
user_input = st.text_area("Paste your email below:")

if st.button("Get Suggestions"):
    suggestions = get_reply_suggestions(user_input)
    for i, suggestion in enumerate(suggestions, 1):
        st.markdown(f"**Suggestion {i}:** {suggestion}")
