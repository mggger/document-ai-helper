import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
import tempfile

from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.streamlit import StreamlitCallbackHandler

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("Document Summarizer & QA")

openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
)

option = st.sidebar.selectbox("Choose your document", options=['default', 'pdf'])

if option == 'default':
    st.info("Please select document type")
    st.stop()

if option == "pdf":
    file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if not file:
    st.info("Please upload pdf file")
    st.stop()

if not openai_api_key:
    st.info("Please set openai api key")
    st.stop()

function_select = st.sidebar.selectbox("Choose the function", options=["default", "summarizer", "chat"])

if function_select == "default":
    st.info("Please select process function")
    st.stop()

llm = ChatOpenAI(openai_api_key=openai_api_key,
                 temperature=0,
                 model_name="gpt-3.5-turbo-16k")

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)


if file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        pdf_path = tmp_file.name
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

    st_cb = StreamlitCallbackHandler(st.container())
    if function_select == "summarizer":
        combined_content = ''.join([p.page_content for p in pages])  # we get entire page data
        texts = text_splitter.split_text(combined_content)
        docs = [Document(page_content=t) for t in texts]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summaries = chain.run(docs, callbacks=[st_cb])
        st.subheader("Summary")
        st.write(summaries)

    elif function_select == "chat":
        if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        user_query = st.chat_input(placeholder="Ask me anything")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                combined_content = ''.join([p.page_content for p in pages])
                texts = text_splitter.split_text(combined_content)
                document_search = Chroma.from_texts(texts, embedding)
                chain = load_qa_chain(llm, chain_type="stuff")
                docs = document_search.similarity_search(user_query)
                response = chain.run(input_documents=docs, question=user_query, callbacks=[st_cb])

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
