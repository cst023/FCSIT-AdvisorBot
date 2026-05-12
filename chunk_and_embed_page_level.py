from dotenv import load_dotenv
import os
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Modify these variables according to your file paths and desired Chroma collection details.
MARKDOWN_FILE = r"C:\Users\christine\Downloads\fcsitadvisorbot_backend\academic_calendar_ug_extracted.md"
PERSIST_DIR = "./chroma_fcsit"
COLLECTION_NAME = "fcsit_unimas_2026" 


# ==============================
# STEP 1 – Load Extracted Markdown File
# ==============================
with open(MARKDOWN_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# ===============================
# STEP 2 – Page-level Chunking
# ===============================

# Split the full text by pages using the "===" marker as a delimiter. Modify this if your markdown uses a different marker to separate pages.
pages = re.split(r"\n===\n", full_text)

documents = []

# ==============================
# STEP 3 – Extract Metadata and Clean Content
# ==============================

for page in pages:
    if "Physical Page" not in page:
        continue

    # Extract metadata
    page_number_match = re.search(r"Physical Page\s+(\d+)", page)
    doc_id_match = re.search(r"Doc ID:\s*(.+)", page)
    source_match = re.search(r"Source:\s*(.+)", page)
    link_match = re.search(r"Link:\s*(.+)", page)
    first_section_title_match = re.search(r"^#\s*(.+)", page, re.MULTILINE)

    page_number = int(page_number_match.group(1)) if page_number_match else None
    doc_id = doc_id_match.group(1).strip() if doc_id_match else None
    source = source_match.group(1).strip() if source_match else None
    link = link_match.group(1).strip() if link_match else None
    first_section_title = first_section_title_match.group(1).strip() if first_section_title_match else "Unknown"


    # Remove metadata text from content
    cleaned_content = re.sub(r"Physical Page\s+\d+", "", page)
    cleaned_content = re.sub(r"Doc ID:.*", "", cleaned_content)
    cleaned_content = re.sub(r"Source:.*", "", cleaned_content)
    cleaned_content = re.sub(r"Link:.*", "", cleaned_content)
    cleaned_content = re.sub(r"Page number:.*", "", cleaned_content)
    cleaned_content = cleaned_content.strip()

    # Create Document object with metadata and cleaned content. The metadata will be useful for retrieval and citation later on. 

    documents.append(
        Document(
            page_content=cleaned_content,
            metadata={
                "page": page_number,
                "doc_id": doc_id,
                "source": source,
                "source_link": link,
                "section": first_section_title
            }
        )
    )


print(f"Created {len(documents)} chunks.") 


# ==============================
# STEP 3 – Initialize Embedding Model
# ==============================

load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_NIM_API")

embedding_model = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2" 
)

# ==============================
# STEP 4 – Create Chroma
# ==============================

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR
)

print("Embedding and storage complete.")

# print the 10th document to verify content and metadata
if len(documents) >= 10:
    print("\nSample Document (10th):")
    print("Metadata:", documents[9].metadata)
    print("Content:", documents[9].page_content[:500], "...")  # Print the first 500 characters of content
