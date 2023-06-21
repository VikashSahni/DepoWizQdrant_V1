import streamlit as st
import csv
import json
import openai
import time
import requests
import PyPDF2
import pandas as pd
import numpy as np
import re
from builtins import PendingDeprecationWarning
# from openai.embeddings_utils import get_embedding

from docx.shared import Inches
import io
from io import BytesIO
import base64
from docx import Document

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client import models
import nltk
from nltk.corpus import stopwords





 
nltk.download('stopwords')
nltk.download('punkt')

# Global Parameters *********************************************

FILE = ''
DIMENSION = 1536  # Embeddings size
# COLLECTION_NAME = 'Deposition_Docs'  # Collection name
QDRANT_HOST = '4.157.146.99'  # Milvus server URI
QDRANT_PORT = '6333'
OPENAI_ENGINE = 'text-embedding-ada-002'  # Which engine to use

openai.api_base = "https://chat-gpt-a1.openai.azure.com/"
openai.api_type = "azure" # manually added by Vikash Sahni
openai.api_key = 'c09f91126e51468d88f57cb83a63ee36'  # Use your own Open AI API Key here
openai.api_version = "2022-12-01" # manually added by Vikash Sahni



# connecting to the server

client = QdrantClient(host="4.157.146.99", port=6333)


# My API Key
api_key = "c09f91126e51468d88f57cb83a63ee36"


token = "50.0"
temp = "0.0"
top_p = "0.90"
f_pen = "0.0"
p_pen = "0.0"


# ***************************************************************


# Functions *****************************************************
def csv_load(file):
    with open(file, newline='',encoding='utf-8') as f:
        reader=csv.reader(f, delimiter=',')
        for row in reader:
            yield row



def embed(text):
    embedding = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002"
    )
    embeddings = embedding['data'][0]['embedding']
    return embeddings


# def embed(text):
#     embedding = openai.Embed.from_prompt(text, engine=OPENAI_ENGINE)
#     return embedding["choices"][0]["embedding"]



def make_valid_partition_name(name):
  # Strip any spaces from the name.
  name = name.strip()

  # Replace any special characters with underscores.
  for char in " .,(){}[]<>*-":
    name = name.replace(char, "_")

  # Check if the name is too long.
  if len(name) > 64:
    st.warning("The file name is too long.")
    exit()

  # Return the valid partition name.
  return name





def extract_pdf_text_to_dataframe(file):
    pdf = open(f"{file}.pdf", 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf)

    extracted_text = []
    page_numbers = []

    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        extracted_text.append(text)
        page_numbers.append(page_num+1)

    data = {'page_no': page_numbers, 'content': extracted_text}
    df = pd.DataFrame(data)
    df_name = make_valid_partition_name(file)

    df.to_csv(f"dataframe_{df_name}.csv", index=False)
    st.success("pdf to dataframe conversion successful")
    # return df_name




def normalize_text(text):
    # Normalize Text
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r". ,", "", text)
    text = text.replace("..", ".")
    text = text.replace(". .", ".")
    text = text.replace("\n", "")
    text = text.strip()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Punctuation Removal
    tokens = [token for token in tokens if re.match(r'\w', token)]

    return " ".join(tokens)




# def append_name_to_file(file_name, name):
#     with open(file_name, 'a') as file:
#         file.write(name + '\n')



def save_to_database(dfName):
    collection = make_valid_partition_name(dfName)
    # append_name_to_file("collection.txt", collection)

    client.recreate_collection(
    collection_name=collection,
    vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE),
    )

    FILE = dfName+".csv"
    
    points =[]

    for idx, row in enumerate(csv_load(FILE)):
        points.append(PointStruct(id=idx, vector=embed(row[1]), payload={"page_no":row[0]}))
    
    operation_info = client.upsert(
        collection_name= collection,
        wait=True,
        points=points
    )

    points = []

    print("Operation info:", operation_info)
    st.success("Dataframe saved to database. Successful")




# def fetch_documents():
#     with open("collection.txt", 'r') as file:
#         names = file.read().splitlines()
#     return names

def fetch_documents():
    collections_response = [name.name for name in client.get_collections().collections]
    return collections_response




def search(query,collection):
    # st.write(collection)
    search_result = client.search(
        collection_name=collection,
        query_vector= embed(normalize_text(query)),
        limit=20,
    )
    # st.write(search_result)
    page_list = []
    for result in search_result:
        # print(result)
        page_list.append(result.payload['page_no'])
    # st.write(page_list)

    return page_list



def get_completions(prompt, temp='0.0',top_p='0.75'):
    url = "https://chat-gpt-a1.openai.azure.com/openai/deployments/DanielChatGPT/chat/completions?api-version=2023-03-15-preview"
    headers = {'Content-Type': 'application/json',
               'api-key': api_key,
               'max_tokens': str(token),
               'temperature': str(temp) if temp else "0.0",
               'top_p': str(top_p) if top_p else "0.75",
               'frequency_penalty': str(f_pen) if f_pen else "0.0",
               'Presence_penalty': str(p_pen) if p_pen else "0.0"
               }
    data = prompt

    body = str.encode(json.dumps(data))

    r = requests.post(url=url, data=body, headers=headers)
    response = r.content

    response = json.loads(response.decode())
    if not ('choices' in response):
        return "Choices Error"
    # print(response)
    response = response["choices"][0]["message"]["content"]
    
    return response



def getAns(pdf_text, que, name):
    data = {'messages': [{"role": "system", "content": f"""You are a deposition QnA bot,and based on the given deposition conversation text, you provide the best answer to the question asked by an attorney. witness name: {name}. Remember everywhere in the answer address the witness by their name. """},
{"role": "user", "content": f"""
You are a Deposition Document QnA bot.
You are provided with the question asked by the attorney and the deposition document conversation text delimited by ``` .
Your task is to review the deposition text and based on you creativity and understanding you have to provide the best possible answer to the given question.
Below each answer mention the page number and line number of the discussion enclosed in parenthesis ().

Important NOTE: If answer cannot be found then just return *** .

Provided Details:
```
Question by an Attorney: {que}
Deposition document text: {pdf_text}```
"""}]}
    
# If the answer cannot be found in Depostion document text then just return "NA".
# (NOTE: If you couldnt find the answer in given text then just return " NA ".)

    timer = 10
    for _ in range(3):
        try:
            answers = get_completions(data)
            break
        except Exception as e:
            st.write(f"Retrying after {timer} seconds...")
            time.sleep(timer)
            timer += 10
    return answers



def summarizeAns(loa,query):
    data = {'messages': [{"role": "system", "content": "You are a deposition QnA bot,and based on the given list of answers and a query you create the detailed answer to the query asked by an attorney."},
{"role": "user", "content": f"""
You are a Deposition Document QnA bot.
You are provided with the query asked by the attorney and the list of answers generated by GPT delimited by ``` .
Your task is to review the answers and query and based on you creativity and understanding you have to create the best possible detailed answer to the given query.
Provided Details:
```
Query by an Attorney: {query}
List of answers: {loa}```
"""}]}

    ans = get_completions(data)
    return ans




def searchDf(listOfPgno, qnaDf, question, name):
    # st.write(listOfPgno)
    content_list = []
    answer = []
    
    for pageno in listOfPgno:
        index = pageno - 1  # Convert pageno to dataframe index
        content_list.extend(qnaDf.loc[index:index+4, 'content'].tolist())
        ans = getAns(content_list, question, name)
        content_list = []

        # st.write(len(ans))
        if len(ans)<200 or "***" in ans:
            continue
        # st.write(len(ans))

        st.write(f"Page no.{pageno}:")
        st.write(ans)
        answer.append(f"""Page no.{pageno}:\n\n {ans} \n\n ************************************** \n""")
        st.write("**************************************")
    
    return answer

def format_list(lst):
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i])
        if i+2 < len(lst) and lst[i]+2 == lst[i+2]:
            i += 2
        else:
            i += 1
    return result



def request_details(text):
    prompt = {'messages': [{"role": "system", "content": "You are a helper function, you will be provided with the text from deposition document. Your task is to extract and return the witness's name."},
                         {"role": "user", "content": f"""
You are provided the deposition text delimited by ```.
Your task is to perform the following actions:
1. Extract the Full name of the witness.
Deposition text: ```{text}```
"""}]}

    details = get_completions(prompt)
    return details

def extract_name_from_csv(df):
    content_list = df['content'].head(5).tolist()
    name = request_details(content_list)
    return name

# ***************************************************************




def app2():

    # Sidebar
    # drop = st.sidebar.button("Drop Collection")
    # if drop:
    #     collectionsList = fetch_documents()
    #     for collection in collectionsList:
    #         client.delete_collection(collection_name=collection)
    #         with open('collection.txt', 'w') as file:
    #             file.truncate(0)
    #     st.success(f"The file collections has been cleared.")


    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Document for QnA")
    document_list = st.sidebar.selectbox("Documents", fetch_documents())
    # save_preferences = st.sidebar.button("Save Preference")
    # st.write(document_list)




    # Main screen
    st.title("Talk with your document")
    # st.subheader("Talk to Document: Question and Answering over Document")

    with st.expander("Follow these steps to use the app"):
        # st.markdown(":")
        # st.markdown("1. Upload the document in Sidebar uploader")
        st.markdown("1. Select the documents for QnA in sidebar")
        st.markdown("2. Search for a query and get response")
        # Add more tutorial steps as needed

    que = st.text_input("Enter your query or question")
    get_response_button = st.button("Get Response")


    if get_response_button and que:
        # utility.drop_collection(COLLECTION_NAME)
        pglist = search(que,document_list)
        loi = []
        for integer in pglist:
            if integer.isdigit():
                loi.append(int(integer))
        loi.sort()
        loi = format_list(list(set(loi)))
        # st.write(loi)
        # print(loi)
        # import the CSV file as a dataframe
        st.write(f"Selected Document: {document_list}")
        qnaDf = pd.read_csv(f"dataframe_{document_list}.csv")

        witness_name = extract_name_from_csv(qnaDf)
        # st.write(witness_name)
        
        
        answer = searchDf(loi, qnaDf, que, witness_name)


        st.write(que)
        ans = summarizeAns("\n".join(answer),que)
        st.write("### Summarized Answer")
        st.write(ans)

        document = Document()

        answers = "\n\n".join(answer)

        # Add heading to the document
        heading = document.add_heading("Generated Response", level=1)
        document.add_paragraph(f"Question: {que}")
        heading2 = document.add_heading("Answers", level=2)
        document.add_paragraph(answers)
        heading2 = document.add_heading("Summarized Answer", level=2)
        document.add_paragraph(ans)
        

        # Save the document as a bytes object
        doc_bytes = io.BytesIO()
        document.save(doc_bytes)
        doc_bytes.seek(0)
        # Download the document as a file
        b64 = base64.b64encode(doc_bytes.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="QNA_{document_list}.docx">Download QNA Docx</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Handle getting response for the query
