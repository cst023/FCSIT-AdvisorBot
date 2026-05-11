from dotenv import load_dotenv
import os
import time
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

# ==============================
# LOAD VECTORSTORE
# ==============================

PERSIST_DIR = "./chroma_fcsit"
COLLECTION_NAME = "fcsit_unimas_2026"

load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_NIM_API")

embedding_model = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2"
)

#alternative:
#nvidia/llama-3.2-nv-embedqa-1b-v2 (deprecating on 2026-05-20)


vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)


# ============================
# FORMAT RETRIEVED DOCUMENTS
# ============================

def format_docs(docs):

    formatted_context = ""
    retrieved_page_doc_ids = []

    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        doc_id = doc.metadata.get("doc_id", "Unknown")
        source = doc.metadata.get("source", "Unknown")
        link = doc.metadata.get("source_link", "Unknown")
        retrieved_page_doc_ids.append(f"{page}-{doc_id}")

        formatted_context += f"""
Source: {source}
Doc ID: {doc_id}
Page: {page}
Link: {link}

Content:
{doc.page_content}

---
"""

    print(f"Retrieved pages: {', '.join(retrieved_page_doc_ids)}")

    return formatted_context


# ============================
# LLM
# ============================

llm = ChatNVIDIA(
    model="openai/gpt-oss-120b", 
    temperature=0.3,
    top_p=0.7,
    max_completion_tokens=1500,
)

# alternative:
# openai/gpt-oss-20b

intent_classifier_llm = ChatNVIDIA( 
    model="google/gemma-2-2b-it",
    temperature=0.2,
    max_completion_tokens=100,
  )


# ============================
# INTENT CLASSIFIER
# ============================

intent_prompt = ChatPromptTemplate.from_messages([
    (
        "user",
        """
You are an intent classifier for an academic advising chatbot for
Faculty of Computer Science and Information Technology (FCSIT) UNIMAS.

Classify the user query into ONE of these categories: "academic_query", "greeting", "thanks".

academic_query
- Questions about faculty/university information
- programme structure
- courses
- grading system
- credit transfer
- curriculum requirements
- handbook policies

greeting
- greetings or casual conversation

thanks
- expressions of gratitude

Return ONLY the category name.

User question: {question}
"""
    )
])

intent_chain = (
    intent_prompt
    | intent_classifier_llm
    | StrOutputParser()
)


# ============================
# RAG PROMPT
# ============================

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "user",
        """
You are an academic advising assistant for undergraduate students from the Faculty of Computer Science and Information Technology (FCSIT), University Malaysia Sarawak (UNIMAS).

Answer the user's question ONLY using the provided context.

Rules:

1. Do NOT use knowledge outside the context to answer academic-related queries. The context is where we can find information specific to FCSIT UNIMAS.
2. If the answer cannot be found, say:

"I am unable to find relevant information to answer your query.
Please refer to your FCSIT handbook or consult your academic advisor."

Do NOT include source citations in this case.

3. Cite the source and page number in your answer. (eg. Source: Page 10, UNIMAS FCSIT Handbook 2025/2026), and cite the source link in your answer. (eg. Link: https://sourcelink.com)
4. Be helpful, accurate and ethical. Your help is greatly appreciated.
5. When faced with case-specific queries, provide general guidance based on the handbook content without giving specific advice that may require human judgment. Direct the user to consult their academic advisor for personalized advice.
6. When user ask for help to calculate their GPA/CGPA, give basic guidance on how the GPA/CGPA is calculated only if such information is found in your context, and direct them to use the GPA/CGPA calculator tool found in the app menu of this FCSIT AdvisorBot mobile app.
7. Present information in a clear and concise textual format. Avoid using tables, as the mobile app interface may not display tables properly.

Context:
{context}

User question: {question}

Conversation so far (summary):
{conversation_summary}
"""
    )
])


summarize_prompt = ChatPromptTemplate.from_messages([
    (
        "user",
        """
Progressively summarize this academic advising conversation.
Add the new exchange to the existing summary and return an updated summary.
Be concise. Focus on topics discussed, courses mentioned, and key facts established.

Existing summary:
{existing_summary}

New exchange:
Human: {question}
AI: {answer}

Updated summary:
"""
    )
])

summarize_chain = summarize_prompt | intent_classifier_llm | StrOutputParser()


# ============================
# RETRIEVER
# ============================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # k=5, retrieve top 5 most similar results
)


# ============================
# VECTOR RAG CHAIN
# ============================

rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "conversation_summary": itemgetter("conversation_summary")
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ============================
# GREETING HANDLER
# ============================

def handle_greeting():

    return """
Hello! 👋

I am AdvisorBot, an academic advising chatbot for FCSIT UNIMAS.

You can ask me general academic questions about FCSIT UNIMAS, such as:

• programme structure

• course requirements

• grading system

• credit transfer policies

• faculty information

How can I assist you today?
"""


# ============================
# ROUTER
# ============================

def route_query(user_query, conversation_summary=""):

    intent = intent_chain.invoke({"question": user_query}).strip().lower()

    if intent == "academic_query":
        answer = rag_chain.invoke(
            {"question": user_query,
             "conversation_summary": conversation_summary
            })   

    elif intent == "greeting":
        answer = handle_greeting()

    elif intent == "thanks":
        answer = "You're welcome! If you have any more questions about FCSIT UNIMAS, feel free to ask."

    else:
        answer = """
I can only assist with questions related to FCSIT UNIMAS academic advising.

Please ask about programme structure, courses, grading system,
or other handbook-related topics. 
"""
    updated_summary = summarize_chain.invoke(
        {
            "existing_summary": conversation_summary,
            "question": user_query,
            "answer": answer
        }
    )     
    return {"answer": answer, "updated_summary": updated_summary}


def process_query_with_timing(user_query, conversation_summary=""):
    start_time = time.perf_counter()
    result = route_query(user_query, conversation_summary)
    elapsed_seconds = time.perf_counter() - start_time
    return result, elapsed_seconds


# ============================
# INTERACTIVE LOOP
# ============================
if __name__ == "__main__":
    conversation_summary = ""
    while True:

        user_query = input("Enter your question (or -1 to exit): ")

        if user_query.strip() == "-1":
            print("Exiting. Goodbye!")
            break

        result, elapsed_seconds = process_query_with_timing(user_query, conversation_summary)
        conversation_summary = result["updated_summary"]

        print(f"\nAdvisorBot:\n{result['answer']}\n")
        print(f"Time taken: {elapsed_seconds:.2f} seconds\n")
