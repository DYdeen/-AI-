import os
import pandas
import pandas as pd
import numpy as np
from ast import literal_eval
import dashscope
from dashscope import TextEmbedding

from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# %% MySQL 数据导入部分
import mysql.connector  # 使用 mysql.connector 而不是 mysql


def read_csv(path):
# 读取CSV数据并转换嵌入数据
    dataframe = pandas.read_csv(path)
    return dataframe


def embedding_vector(dataframe):
# %%Create embeddings
# text_type=`document` to bui  ld index
    embedder = DashScopeEmbedding(
     model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
     text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
     )
    text_to_embedding = dataframe.iloc[:, 0]
    # Call text Embedding
    result_embeddings = embedder.get_text_embedding_batch(text_to_embedding)
    return result_embeddings

def faiss_vector(result_embeddings):
    # create a faiss index
    result_embeddings_np = np.array(result_embeddings, dtype="float32")
    d = 1536  # dimension
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(result_embeddings_np)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # 定义存储路径
    output_dir = "./data/faiss/"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    persist_path = os.path.join(output_dir, "faiss_index.index")
    # 保存索引到磁盘
    vector_store.persist(persist_path=persist_path)
    print(f"Faiss索引已保存到 {persist_path}")

def create_table(connection,schema_file_path):
    with open(schema_file_path, 'r', encoding='utf-8') as schema_file:
         sql_commands= schema_file.read().strip(";")#去分号并拆句

    with connection.cursor() as cursor:
        for command in sql_commands:
            command = command.strip()  # 去除空白字符
            if command:
                try:
                    # 如果是 CREATE TABLE 语句，可以加入判断，跳过表已存在的错误
                    if command.lower().startswith("create table"):
                        # 使用 IF NOT EXISTS 来避免创建已存在的表
                        if "if not exists" not in command.lower():
                            command = command.replace("create table", "create table if not exists")
                        cursor.execute(command)
                        print(f"成功执行表创建命令: {command[:50]}")
                    else:
                        continue  # 只执行 CREATE TABLE，不执行其他语句
                except mysql.connector.Error as e:
                    if e.errno == 1050:  # 表已存在错误
                        print(f"表已存在，跳过语句: {command[:50]}")
                    else:
                        raise
        connection.commit()
    print(f"已执行表结构创建 SQL 文件：{schema_file_path}")

def insert(connection, insert_sql_file_path):

    with open(insert_sql_file_path, "r", encoding="utf-8") as f:
        sql_commands = f.read().split(";")  #去分号并拆句

    with connection.cursor() as cursor:
        for command in sql_commands:
            command = command.strip()  # 去除空白字符
            if command:
                try:
                    # 检查是否是 INSERT INTO 语句
                    if command.lower().startswith("insert into"):
                        # 提取 VALUES 部分的内容
                        values_part = command.split("VALUES")[1].strip(" ()")  # 去除括号
                        text_value = values_part.strip("'")  # 去除单引号

                        # 检查数据库中是否已有相同的 text
                        cursor.execute("SELECT COUNT(*) FROM ai_context WHERE text = %s", (text_value,))
                        count = cursor.fetchone()[0]

                        if count == 0:  # 如果数据库中没有相同的 text，则执行插入
                            cursor.execute(command)
                            print(f"成功执行插入命令: {command[:200]}")
                        else:
                            print(f"插入跳过：text '{text_value}' 已存在。")
                    else:
                        continue  # 只执行 INSERT INTO，不执行其他 SQL
                except mysql.connector.Error as e:
                    print(f"执行插入命令失败: {command[:50]}，错误：{e}")
                    raise
        connection.commit()
    print(f"已执行插入数据 SQL 文件：{insert_sql_file_path}")

# Main
# 设置DashScope的API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
datafile_path = "运动鞋店铺知识库.txt"
df = read_csv(datafile_path)
result_embeddings=embedding_vector(df)
faiss_vector(result_embeddings)
# 数据库连接信息
db_host = "localhost"
db_user = "root"  # 数据库用户名
db_password = os.getenv("DB_PASSWORD")  # 数据库密码
db_name = "rag_database"  # 数据库名称
# 连接到 MySQL 数据库
connection = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
)
# 执行 SQL 文件中的建表语句
schema_file_path = "project2_RAG/schema.sql"
create_table(connection, schema_file_path)

# 执行 SQL 文件中的插入数据语句
insert_sql_file_path = "project2_RAG/ai_context.sql"  # 请确认文件名为 ai_context.sql
insert(connection, insert_sql_file_path)

# 关闭数据库连接
connection.close()
print("连接已断开")
