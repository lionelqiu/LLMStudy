import pandas as pd
import openai
from openai.embeddings_utils import cosine_similarity
import ast
import streamlit as st


df = pd.read_csv('qa_dataset_with_embeddings.csv')

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Convert the string embeddings back to lists
df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)

st.title("💬 Smart FAQ Assistant")

user_question = st.text_input("Ask a question:")
col1, col2 = st.columns([1, 1])
search_clicked = col1.button("🔍 Search")
clear_clicked = col2.button("❌ Clear")

if clear_clicked:
    st.experimental_rerun()

if search_clicked and user_question.strip():

    # Get embedding for the user's question
    user_question_embedding = get_embedding(user_question)

    # Calculate cosine similarities for all questions in the dataset
    df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

    # Find the most similar question and get its corresponding answer
    most_similar_index = df['Similarity'].idxmax()
    max_similarity = df['Similarity'].max()

    # Set a similarity threshold to determine if a question is relevant enough
    similarity_threshold = 0.85  # You can adjust this value
    
    if max_sim >= threshold:
        st.success(f"**Answer:** {df['answer'][most_similar_index]}")
        st.caption(f"🧠 Similarity Score: {max_similarity:.2f}")
        
        st.write("Was this helpful?")
        if st.button("👍 Yes"):
            st.success("Thanks for your feedback!")
        if st.button("👎 No"):
            st.warning("We'll work on improving the answer.")
    else:
        st.warning("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")

# Optional: FAQ Section
with st.expander("📚 Common Questions"):
    for q in df['question']:
        st.markdown(f"- {q}")
