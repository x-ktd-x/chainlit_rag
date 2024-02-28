import os
import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# プロンプト定義
prompt = PromptTemplate(
    template="""
    文章を前提にして質問に答えてください。

    文章 :
    {document}

    質問 : {question}
    """,
    input_variables=["document", "question"],
)

@cl.on_chat_start
async def on_chat_start():
    """初回起動時に呼び出される."""
    global files  # グローバル変数として扱うために宣言
    files = None

    # awaitメソッドのために、whileを利用する。アップロードされるまで続く。
    while files is None:
        # chainlitの機能に、ファイルをアップロードさせるメソッドがある。
        files = await cl.AskFileMessage(
            max_size_mb=20,  # ファイルの最大サイズ
            content="PDFを選択してください。",  # ファイルをアップロードさせる画面のメッセージ
            accept=["application/pdf"],  # PDFファイルを指定する
            raise_on_timeout=False,  # タイムアウトなし
        ).send()

    file = files[0]
    # アップロードされたファイルのパスから中身を読み込む。
    documents = PyMuPDFLoader(file.path).load()

    text_splitter = SpacyTextSplitter(chunk_size=400, pipeline="ja_core_news_sm")
    splitted_documents = text_splitter.split_documents(documents)

    # テキストをベクトル化するOpenAIのモデル
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Chromaにembedding APIを指定して、初期化する。
    database = Chroma(embedding_function=embeddings)

    # PDFから内容を分割されたドキュメントを保存する。
    database.add_documents(splitted_documents)

    # 今回は、簡易化のためセッションに保存する。
    cl.user_session.set("data", database)

    await cl.Message(content="チャット開始").send()


@cl.on_message
async def on_message(input_message: cl.Message):
    """メッセージが送られるたびに呼び出される."""

    # チャット用のOpenAIのモデル
    open_ai = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

    # セッションからベクトルストアを取得（この中にPDFの内容がベクトル化されたものが格納されている）
    database = cl.user_session.get("data")

    # 質問された文から似た文字列を、DBより抽出
    documents = database.similarity_search(input_message.content)

    # 抽出したものを結合
    documents_string = ""
    for document in documents:
        documents_string += f"""
        ---------------------------------------------
        {document.page_content}
        """

    # プロンプトに埋め込みながらOpenAIに送信
    result = open_ai(
        [
            HumanMessage(
                content=prompt.format(
                    document=documents_string,
                    question=input_message.content  # 'query'を'question'に修正
                )
            )
        ]
    ).content

    # 下記で結果を表示する(content=をOpenAIの結果にする。)
    await cl.Message(content=result).send()
