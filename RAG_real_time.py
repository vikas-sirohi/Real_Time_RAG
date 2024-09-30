import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


load_dotenv()

# I have used Groq mixtral LLM you can use yours. Save your API in .env File.

current_dir = os.getcwd()
# ---- VectorDB Path ----#
#--- If not exists run 1a_rag_embedd.py ---#
persist_dir = os.path.join(current_dir, "db", "chroma_db")

#---- Embeddings Model ----#
model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name = model_name) 

db = Chroma(persist_directory=persist_dir, embedding_function = hf)

retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":2},
)


llm = ChatGroq(model = "mixtral-8x7b-32768",
                 temperature = 0,
                 max_tokens = 100,
                 max_retries = 2,
                 )

# Our plan is (query_given+llm) -> contexualize the query
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Creating a prompt template for contextualizing questions

contextualize_query_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

# It returns a stroutput query which is aware of the past
history_aware_retriever = create_history_aware_retriever(llm, retriever ,contextualize_query_prompt)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def Chat():
    print("\nStart chatting with the AI! Type 'exit' to end the conversation.\n")
    chat_history = []

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({"input":query,
                                   "chat_history":chat_history})
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content = query))
        chat_history.append(SystemMessage(content = result["answer"]))

if __name__ == "__main__":
    Chat()