RAG Project Structure:
 
E-Commerce-rag/
├── ingest.py
├── data/
│   └── walmart-products.csv
├── models/
│   └── all-MiniLM-L6-v2/
|       ├── config.json
|       ├── pytorch_model.bin
|       ├── tokenizer.json
|       ├── modules.json
|       └── sentence_bert_config.json
├── rag_pipeline.py
├── database.py
├── models.py
├── config.py
|
├── embeddings/
│   └── faiss_index
│
├── requirements.txt
├── README.md
└── .env