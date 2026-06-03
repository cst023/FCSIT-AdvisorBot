from dotenv import load_dotenv
import os
import time
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
import re


# ==============================
# CONFIGURATION
# ==============================

PERSIST_DIR = "./chroma_fcsit"
COLLECTION_NAME = "fcsit_unimas_2026"


# ==============================
# LOAD VECTORSTORE
# ==============================

load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_NIM_API")

embedding_model = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2"
)

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
    model="nvidia/nemotron-3-super-120b-a12b", 
    temperature=0.2,
    top_p=0.95,
    max_tokens=2000,
  
    chat_template_kwargs={"enable_thinking":True,"low_effort":True}
    
)

'''
    alternative model:
    model="openai/gpt-oss-120b", 
    temperature=0.3,
    top_p=0.7,
    max_completion_tokens=1500,

    model="nvidia/nemotron-3-super-120b-a12b",
    temperature=0.2,
    top_p=0.95,
    max_tokens=4000,
  
    chat_template_kwargs={"enable_thinking":True,"low_effort":True}
'''

intent_classifier_llm = ChatNVIDIA( 
    model="google/gemma-2-2b-it",
    temperature=0.2,
    max_completion_tokens=1024,
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

Classify the user query into ONE of these categories: "academic_query", "greeting", "thanks", "follow_up".

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

follow_up
- Messages that need earlier conversation context to understand
- Vague questions like "what about that", "how about this", "what is her email"
- Questions that refer to unspecified people, things, or previous turns
- Very short or ambiguous messages that cannot stand on their own

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

"""
    )
])

# ============================
# RETRIEVER
# ============================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# ============================
# VECTOR RAG CHAIN
# ============================

rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ============================
# FOLLOW-UP HANDLER
# ============================

FOLLOW_UP_RESPONSE = """
I can help best when your question is clear, direct, and self-contained.

I currently cannot handle follow-up questions or messages that depend on earlier chat context.
This is a feature planned for future improvements.

Please rephrase your question in a concise way so I can help you better.
""".strip()


FOLLOW_UP_PATTERNS = [
    r"^\s*what about (that|this|it|him|her|them)\s*\??$",
    r"^\s*how about (this|that|it|him|her|them)\s*\??$",
    r"^\s*what is (her|his|their|its)\b.*\??$",
    r"^\s*what about (it|that|this)\b.*\??$",
    r"^\s*(this|that|it)\s*\??$",
]


def looks_like_follow_up(user_query):
    normalized_query = user_query.strip().lower()

    if any(re.search(pattern, normalized_query) for pattern in FOLLOW_UP_PATTERNS):
        return True

    return False


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

To help me give you the most accurate info, please ensure your questions are:

• Clear and specific

• Concise (avoid unnecessary details)

• Self-contained (avoid vague follow ups)

How can I assist you today?
"""


# ============================
# ROUTER
# ============================

def route_query(user_query):

    intent = intent_chain.invoke({"question": user_query}).strip().lower()

    if intent == "academic_query" and looks_like_follow_up(user_query):
        intent = "follow_up"

    if intent == "academic_query":
        answer = rag_chain.invoke(
            {"question": user_query
            })   

    elif intent == "greeting":
        answer = handle_greeting()

    elif intent == "thanks":
        answer = "You're welcome! If you have any more questions about FCSIT UNIMAS, feel free to ask."

    elif intent == "follow_up":
        answer = FOLLOW_UP_RESPONSE

    else:
        answer = """
I can only assist with questions related to FCSIT UNIMAS academic advising.

Please ask about programme structure, courses, grading system,
or other handbook-related topics. 
"""

    return {
        "answer": answer,
        "intent": intent,
    }


def process_query_with_timing(user_query):
    start_time = time.perf_counter()
    result = route_query(user_query)
    elapsed_seconds = time.perf_counter() - start_time
    return result, elapsed_seconds


# ============================
# INTERACTIVE LOOP
# ============================
if __name__ == "__main__":
    while True:

        user_query = input("Enter your question (or -1 to exit): ")

        if user_query.strip() == "-1":
            print("Exiting. Goodbye!")
            break

        result, elapsed_seconds = process_query_with_timing(user_query)

        print(f"\nAdvisorBot:\n{result['answer']}\n")
        print(f"Time taken: {elapsed_seconds:.2f} seconds\n")
