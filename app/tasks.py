import json
import os

from rq import Queue
from app.redis_client import redis_client
from app.load_data import load_split_pdf_file, initialize_splitter
from app.vector_db import update_vector_store


def process_pdf(pdf_id, pdf_data):
    pdf_data = redis_client.get(pdf_id)
    pdf_data = json.loads(pdf_data)
    for pdf in pdf_data["pdf_list"]:
        pdf_file = f"pdfs/{pdf}"
        if os.path.exists(pdf_file):
            print(f"Processing {pdf}")
        text_splitter = initialize_splitter(1000, 48)
        documents = load_split_pdf_file(pdf_file, text_splitter)
        print(f"Loaded {len(documents)} documents")
        update_vector_store(documents)
    with redis_client.pipeline() as pipe:
        pipe.multi()
        pdf_data["status"] = "done"
        pipe.set(pdf_id, json.dumps(pdf_data))
        pipe.execute()


queue = Queue(connection=redis_client.connect())
