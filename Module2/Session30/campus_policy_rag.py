import os
import re
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# ----------------------------
# Configuration
# ----------------------------

PDF_FOLDER = "Policy_documents"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "campus_policies"

CHUNK_SIZE = 150       # words
OVERLAP_PERCENT = 0.15

api_key = os.getenv("OPENAI_API_KEY")

print("API key loaded:", bool(api_key))

client = OpenAI(api_key=api_key)

chroma_client = chromadb.PersistentClient(path=DB_PATH)


# ----------------------------
# Infer policy type
# ----------------------------

def infer_policy_type(filename):

    name = filename.lower()

    if "hostel" in name:
        return "hostel"

    elif "refund" in name:
        return "refund"

    elif "library" in name:
        return "library"

    return "general"


# ----------------------------
# Clean extracted text
# ----------------------------

def clean_text(text):

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------
# PDF Loader
# ----------------------------

def load_all_pdfs(folder):

    all_docs = []

    pdf_files = Path(folder).glob("*.pdf")

    for file in pdf_files:

        reader = PdfReader(file)

        print(
            f"Loaded {len(reader.pages)} pages from: {file.name}"
        )

        policy_type = infer_policy_type(file.name)

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

                text = clean_text(text)

                all_docs.append({

                    "text": text,
                    "source": file.name,
                    "page": page_num + 1,
                    "policy_type": policy_type
                })

    return all_docs


# ----------------------------
# Chunking
# ----------------------------

def split_chunks(text,
                 chunk_size=CHUNK_SIZE,
                 overlap_percent=OVERLAP_PERCENT):

    words = text.split()

    overlap = int(
        chunk_size * overlap_percent
    )

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunks.append(
            " ".join(chunk_words)
        )

        start += chunk_size - overlap

    return chunks


# ----------------------------
# Embeddings
# ----------------------------

def generate_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ----------------------------
# Build vector database
# ----------------------------

def build_knowledge_base():

    try:

        chroma_client.delete_collection(
            COLLECTION_NAME
        )

    except:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME
    )

    docs = load_all_pdfs(
        PDF_FOLDER
    )

    total_chunks = 0

    for doc in docs:

        chunks = split_chunks(
            doc["text"]
        )

        total_chunks += len(chunks)

        for chunk in chunks:

            embedding = generate_embedding(
                chunk
            )

            collection.add(

                ids=[str(uuid.uuid4())],

                documents=[chunk],

                embeddings=[embedding],

                metadatas=[{

                    "source":
                    doc["source"],

                    "page":
                    doc["page"],

                    "policy_type":
                    doc["policy_type"]
                }]
            )

    print(
        f"Total chunks created: {total_chunks}"
    )

    print(
        f"Successfully stored "
        f"{total_chunks} chunks in "
        f"vector database."
    )

    return collection


# ----------------------------
# Retrieval
# ----------------------------

def retrieve_chunks(query,
                    top_k=3):

    collection = chroma_client.get_collection(
        COLLECTION_NAME
    )

    query_embedding = generate_embedding(
        query
    )

    result = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=top_k
    )

    print(
        f"Retrieved {top_k} relevant chunks."
    )

    return result


# ----------------------------
# Prompt Builder
# ----------------------------

def build_prompt(
        question,
        retrieved_text):

    prompt = f"""

You are a campus policy assistant.

Rules:

1. Answer only from context

2. If answer not present say:

"I don't have that information."

3. Keep response simple
and student-friendly.


Context:

{retrieved_text}


Question:

{question}


Answer:

"""

    return prompt


# ----------------------------
# LLM Answer
# ----------------------------

def generate_answer(prompt):

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return (
        response
        .choices[0]
        .message.content
    )


# ----------------------------
# End-to-end QA
# ----------------------------

def answer_question(question):

    retrieved = retrieve_chunks(
        question
    )

    docs = retrieved["documents"][0]

    context = "\n\n".join(docs)

    prompt = build_prompt(
        question,
        context
    )

    answer = generate_answer(
        prompt
    )

    return answer


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    collection = build_knowledge_base()

    print(
        f"\nVector DB ready. "
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    test_queries = [

        "Can I get a refund after dropping a course?",

        "What is the deadline for returning a library book?",

        "Are hostel visitors allowed on weekends?"
    ]

    for q in test_queries:

        print("\n"+"="*50)

        print(
            f"\nUser Query: {q}"
        )

        answer = answer_question(
            q
        )

        print(
            f"\nAnswer: {answer}"
        )