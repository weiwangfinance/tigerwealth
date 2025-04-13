import streamlit as st
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit.components.v1 import html

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
 
import os

SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'



SPARKAI_APP_ID = os.getenv("SPARKAI_APP_ID")
SPARKAI_API_SECRET = os.getenv("SPARKAI_API_SECRET")
SPARKAI_API_KEY = os.getenv("SPARKAI_API_KEY")

 
 
 
# 加载文件夹中的所有txt类型的文件
#loader = DirectoryLoader('./documents/', glob='*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
#documents = loader.load()
# 初始化加载器
#text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
#split_docs = text_splitter.split_documents(documents)
# Load embedding model 
#embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
#docsearch = Chroma.from_documents(split_docs, embed_model, persist_directory="./vector_store")
#docsearch.persist()
# 加载数据
#docsearch = Chroma(persist_directory="./vector_store", embedding_function=embed_model)



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
        bot_role = """
                   你是一个金融咨询专家，你现在要接受公众的咨询。
                   你可以回答金融问题，养老问题，以及相关的法律问题。
                   你的回答应该从提示人们注意风险，特别是诈骗风险，并给予法律上的建议。
                   但凡不相关的问题，你会说对不起,我只回答与金融与反诈骗相关的咨询。
                    """
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
        #similarDocs = docsearch.similarity_search(user_input, k=1)
        #info = ""
        #for similardoc in similarDocs:
        #    info = info + similardoc.page_content
        #question = "结合以下信息：" + info + "回答" + user_input
        question =  user_input
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



    
disclaimer_text = """
关于[大语言模型名称]测试版的重要提示

亲爱的用户：

欢迎您使用我们的[大语言模型名称]测试版。大语言模型是一项前沿技术，虽然我们致力于为您提供高质量、准确的服务，但由于本模型目前处于测试阶段，存在一定局限性和风险，在此特别向您说明：
1. **模型结果的不确定性**：本测试版模型可能会生成不准确、不完整、有误导性或存在偏差的信息。尤其在处理复杂专业问题、需要精确事实性知识以及对时效性要求极高的内容时，出错的可能性更大。例如在医学领域，对于疾病诊断和治疗方案的建议，模型给出的内容可能无法作为专业医疗指导；在法律事务上，有关法律条文解读和案件处理建议等，模型输出不一定符合实际法律规定和司法实践。
2. **潜在有害内容**：尽管我们已尽力对模型进行优化和过滤，但仍可能出现生成包含歧视性、攻击性、不当、违法或违背道德伦理的内容的情况。比如在涉及不同种族、性别、宗教等话题讨论时，模型可能产生不恰当言论。
3. **隐私与安全风险**：虽然我们采取了一系列措施保护您的输入信息安全，但在使用过程中，仍存在因技术漏洞等不可预见原因导致信息泄露的风险。同时，模型也可能受到外部恶意攻击，如提示注入攻击，导致其行为异常，产生不可控输出。

为了最大程度保障您的权益，避免可能的风险，我们强烈建议您：
1. 对于重要决策、专业事务（如医疗、法律、金融投资等），切勿仅依赖本模型提供的建议，务必寻求专业人士的意见和指导。例如，在制定投资策略时，不能仅凭模型推荐就进行大额资金投入，而应咨询专业金融顾问并结合自身财务状况和风险承受能力。
2. 当发现模型生成的内容存在错误、不当或可能对您造成误导时，请及时停止使用相关内容，并向我们反馈，反馈渠道为[具体反馈方式，如邮箱地址、反馈表单链接等] 。
3. 在输入信息时，请勿提供敏感、机密或您不希望公开的个人信息，如身份证号、银行卡号、商业机密数据等。

特别声明：本模型仅作为测试和探索性工具提供给您使用。您明确知悉并理解模型存在的上述风险，并自愿承担因使用本模型而可能产生的一切后果。对于因使用本测试版模型而导致的任何直接或间接损失、损害，包括但不限于经济损失、数据丢失、名誉损害等，我们不承担任何法律责任。同时，我们保留随时修改、暂停或终止本测试版服务的权利，且无需事先通知。

感谢您对我们工作的理解与支持，您的反馈对我们优化模型至关重要，期待与您共同推动大语言模型技术的进步。

[网站运营主体名称]
[具体日期] 
    """
# 使用 HTML 和 CSS 减小字体大小
styled_text = f'<div style="font-size: 14px;">{disclaimer_text}</div>'
with st.expander("风险提示"):
    st.markdown(styled_text, unsafe_allow_html=True)