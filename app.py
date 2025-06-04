import pandas as pd
import openai
from openai.embeddings_utils import cosine_similarity
import ast
import streamlit as st

# Load dataset
df = pd.read_csv('qa_dataset_with_embeddings.csv')

# Convert string embeddings back to list format
df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)

# Title
st.title("💬 Smart FAQ Assistant")

# Input field
user_question = st.text_input("Ask a question:")

# Buttons
col1, col2 = st.columns([1, 1])
search_clicked = col1.button("🔍 Search")
clear_clicked = col2.button("❌ Clear")

# Clear input
if clear_clicked:
    st.experimental_rerun()

# Define embedding function
def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Handle question search
if search_clicked and user_question.strip():
    user_question_embedding = get_embedding(user_question)

    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    similarity_threshold = 0.85

    if max_similarity >= similarity_threshold:
        st.success(f"**Answer:** {df['answer'][most_similar_index]}")
        st.caption(f"🧠 Similarity Score: {max_similarity:.2f}")

        st.write("Was this helpful?")
        if st.button("👍 Yes"):
            st.success("Thanks for your feedback!")
        if st.button("👎 No"):
            st.warning("We'll work on improving the answer.")
    else:
        st.warning("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")

# Optional FAQ section
with st.expander("📚 Common Questions"):
    for q in df['Question']:
        st.markdown(f"- {q}")
