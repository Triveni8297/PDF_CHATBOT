import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Directory where the FAISS index is stored
INDEX_DIR = "faiss_index"

load_dotenv()

# Load embeddings and vectorstore
print(f"Loading FAISS index from '{INDEX_DIR}'...")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.load_local(INDEX_DIR, embeddings)

# Initialize LLM with optional system prompt (can be customized)
SYSTEM_PROMPT = """
You are a friendly, conversational AI assistant 😊 specialized in answering questions about generative AI and its applications, based on the content of the provided PDF document.

– 👋 Whenever the user greets you (e.g., “hi,” “hello,” “how are you?”), respond with a warm and natural greeting.  
– 🙂 When the user introduces themselves, welcomes you, or asks about you, reply in a personable, upbeat tone.  
– 🎉 If they offer well wishes (“happy holidays,” “good luck,” etc.) or say goodbye, reply with an appropriate friendly farewell.  
– 📖 For any question directly related to generative AI or the content in the PDF, retrieve relevant context and give a clear, concise, accurate answer.  
– ❗ If the user’s question falls outside generative AI topics, politely say: “I’m sorry, I can only help with questions about generative AI and its applications.”  
– 🤔 If you don’t know the answer from the PDF, say: “I’m not sure about that—let me know if you’d like to ask another question about generative AI.”

Always keep your tone warm, helpful, and professional 🌟.

# Example conversation:
# User: Hi there!  
# Assistant: Hello! How can I assist you today? 😊  
#
# User: How are you?  
# Assistant: I'm doing great, thank you! How about you? 🙂
"""



llm = ChatOpenAI(temperature=0)

# Build a ChatPromptTemplate for the QA chain that accepts both context and question
qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """
Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    )
])

# Persistent conversational memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Build the ConversationalRetrievalChain, injecting our QA prompt and specifying the document variable name
conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": qa_prompt,
        "document_variable_name": "context"
    }
)


def ask_query(user_question):
    # Ask the chain
    result = conversation({"question": user_question})
    answer = result.get("answer") or result.get("response")
    print("**************************************************")
    print(f"User: {user_question}")
    print(f"Assistant: {answer}\n")
    print("**************************************************")
    return answer



if __name__ == "__main__":
    # Example usage
    user_question = "hello"
    response = ask_query(user_question)


