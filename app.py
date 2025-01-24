import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import FireCrawlLoader

# URLs to scrape documents from
urls = [
    "https://consultquark.com/THA_Act_1996.pdf"
]

# Load documents from the URLs
docs = [FireCrawlLoader(api_key="fc-d49c1c3467fb4e15b6713ff0bbb5cbf6", url=url, mode="scrape").load() for url in urls]
# Flatten the loaded documents
docs_list = [item for sublist in docs for item in sublist]

# Streamlit UI elements
st.title("Document Processing with LangChain")
st.write("This app processes PDF documents and splits them into manageable chunks.")

# User input for chunk size and overlap
chunk_size = st.number_input("Chunk size", min_value=1, value=250, step=1)
chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=0, step=1)

# Process documents when the user clicks the button
if st.button("Process Documents"):
    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        # Filter and prepare documents for display
        filtered_docs = []
        for doc in doc_splits:
            if isinstance(doc, Document) and hasattr(doc, 'metadata'):
                clean_metadata = {
                    k: v for k, v in doc.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
                filtered_docs.append({
                    'content': doc.page_content,
                    'chunk': len(doc.page_content)
                })

        # Display the processed chunks
        if filtered_docs:
            st.subheader("Processed Chunks:")
            for idx, doc in enumerate(filtered_docs):
                st.write(f"**Chunk {idx + 1}:**")
                st.write(f"Content: {doc['content'][:500]}...")  # Display a preview of the content
                st.write(f"Chunk size: {doc['chunk']} characters")
                st.markdown("---")
        else:
            st.warning("No chunks were processed.")
    except Exception as e:
        st.error(f"Error processing the documents: {str(e)}")
