import json
import time
from pprint import pprint

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import uuid

from langchain.chains.retrieval_qa.base import RetrievalQA

from app.redis_client import redis_client
from app.models import PDFFile
from app.tasks import process_pdf, queue
from app.loal_llm import load_openllm
from app.vector_db import load_vector_store

app = FastAPI()


@app.post("/upload-files/")
async def create_upload_files(files: list[UploadFile] = File(...)):
    task_id = str(uuid.uuid4())
    pdf_list = []
    try:
        for file in files:
            with open(f'pdfs/{file.filename}', 'wb') as f:
                f.write(file.file.read())
            if file.content_type != "application/pdf":
                return {"message": "Only pdf and html files are supported"}
            if file.filename.endswith('.pdf'):
                pdf_list.append(f'{file.filename}')
        pdf_data = PDFFile(task_id=task_id, pdf_list=pdf_list)
        pdfs_json = json.dumps(pdf_data.__dict__)
        redis_client.set(task_id, pdfs_json)
        # redis_client.connect().hset(f"pdf:{task_id}", mapping={"status": PDFStatus.PROCESSING})
        print(f"PDF file uploaded and split")
        # process_pdf(task_id, pdf_data)
        job = queue.enqueue(process_pdf, task_id, pdf_data)
        return pdf_data
    except Exception as e:
        return {"error": str(e)}


@app.get("/status")
async def check_status(request: Request, pdf_id: str):
    status = redis_client.get(pdf_id)
    if status:
        return json.loads(status)
    else:
        return JSONResponse("Item ID Not Found", status_code=404)


@app.get("/query")
async def query(request: Request, query: str):
    start_time = time.time()
    vector_store = await load_vector_store()
    qa_chain = RetrievalQA.from_chain_type(llm=load_openllm(), retriever=vector_store.as_retriever(), return_source_documents=True)
    result = qa_chain.invoke({"query": query})
    source_documents = []
    for doc in result["source_documents"]:
        source_documents.append({"source": doc.metadata["source"], "page": doc.metadata["page"]})
    end_time = time.time()
    return {"result": result["result"], "response_time": end_time - start_time, "source_documents": source_documents}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app)
