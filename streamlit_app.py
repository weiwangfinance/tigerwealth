import streamlit as st
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
 
import os

SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'



SPARKAI_APP_ID = os.getenv("SPARKAI_APP_ID")
SPARKAI_API_SECRET = os.getenv("SPARKAI_API_SECRET")
SPARKAI_API_KEY = os.getenv("SPARKAI_API_KEY")


 
# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('./documents/', glob='*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()
# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)
# Load embedding model 
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embed_model, persist_directory="./vector_store")
docsearch.persist()
# 加载数据
docsearch = Chroma(persist_directory="./vector_store", embedding_function=embed_model)



# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Function to send a message to Ollama API
def query_ollama(prompt):
    try:
        spark = ChatSparkLLM(
                spark_api_url=SPARKAI_URL,
                spark_app_id=SPARKAI_APP_ID,
                spark_api_key=SPARKAI_API_KEY,
                spark_api_secret=SPARKAI_API_SECRET,
                spark_llm_domain='lite',
                streaming=False,
            )
        bot_role = "你是一个反金融诈骗专家，你现在要接受公众的咨询。但凡不相关的问题，你会说对不起,我只回答与金融与反诈骗相关的咨询。"
        messages = [ChatMessage(role="system",content=bot_role),
            ChatMessage(role="user",content=prompt)]
        #handler = ChunkPrintHandler()
        #response = spark.generate([messages], callbacks=[handler])
        response = spark.generate([messages])
        return response
    except Exception as e:
        return f"Error connecting to Spark API: {str(e)}"


# Streamlit 应用
st.set_page_config(
    page_title="钱袋萌虎(Tiger Wealth)",
    page_icon="📊",
    layout="wide"
)

# 添加 Logo
st.image("logo.png", width=50)  # 确保 logo.png 文件在您的工作目录中


st.title("钱袋萌虎 (Tiger Wealth)")
st.markdown("与小虎反诈助手聊天，获取相关信息和建议。")
st.markdown("Chat with Tiger Wealth, An Anti-Fraud Assistant, to get relevant information and advice!")


# Chat input
user_input = st.text_input("来聊聊天吧/Let's Talk:", placeholder="Type your message here...")



# Process user input
if user_input:
    # Display user's message in the chat
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.spinner("思考中/Thinking..."):
        # 加载之前持久化数据（k值为选取4个最匹配结果输出）
        similarDocs = docsearch.similarity_search(user_input, k=1)
        info = ""
        for similardoc in similarDocs:
            info = info + similardoc.page_content
        question = "结合以下信息：" + info + "回答" + user_input

        # Query Ollama and get response
        bot_response = query_ollama(prompt=question )
        st.session_state.conversation.append({"role": "bot", "content": bot_response.generations[0][0].text})



# Process user input
if user_input:
    with st.spinner("思考中/Thinking..."):
        # Query Ollama and get response
        bot_response = query_ollama(prompt=user_input)
        st.markdown(f"**用户(User):** {user_input}")
        st.markdown(f"**小虎(Tiger):** {bot_response.generations[0][0].text}")
    
