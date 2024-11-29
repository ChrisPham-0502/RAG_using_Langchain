# RAG using Langchain
This project leverages Langchain framework to build RAG pipeline and uses streamlit to create temporary frontend for user interaction. 

Retrieval augmented generation (RAG) is a natural language processing (NLP) technique that combines the strengths of both retrieval- and generative-based artificial intelligence (AI) models.
![1](https://github.com/ChrisPham-0502/RAG_using_Langchain/assets/126843941/8a434ec4-e41f-4109-922a-dfe703fdd832)

## Inference
Clone this repository on your local environment.
```sh
git clone https://github.com/ChrisPham-0502/RAG_using_Langchain.git
```

Then, install some necessary libraries:
```sh
pip install -r requirements.txt
```

  Finally, run the frontend to open chatbot:
```sh
streamlit run RAG.py
```

## Configuration
To successfully run this repo, you need to put your OpenAI API key into **.env file** or directly assign it to a variable in RAG.py: os.environ["OPENAI_API_KEY"]="YOUR KEY"  
