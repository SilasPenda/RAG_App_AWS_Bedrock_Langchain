import streamlit as st

# Vector Embeddings and Vector Store
from langchain.vectorstores import FAISS

from utils.rag_utils import RAG


RAG_helper = RAG()


def main():
    st.set_page_config("PDF Q&A")
    st.header("PDF Q&A using AWS Bedrock 💁")

    user_question = st.text_input("Ask question from PDF Files")

    with st.sidebar:
        st.title("Update or Create Vectore Store:")

        # File uploader widget
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_file is not None:
            # Save the uploaded file
            RAG_helper.save_uploadedfile(uploaded_file)

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = RAG_helper.data_ingestion()
                RAG_helper.get_vector_store(docs)
                st.success("Done!")

    # if st.button("Claude Output"):
    #     with st.spinner("Processing..."):
    #         faiss_index = RAG_helper.load_vectors()
    #         llm = RAG_helper.get_claude_llm()

    #         st.write(RAG_helper.get_response_llm(llm, faiss_index, user_question))
    #         st.success("Done!")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = RAG_helper.load_vectors()
            llm = RAG_helper.get_llma2_llm()
            
            st.write(RAG_helper.get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


if __name__ == "__main__":
    main()

        