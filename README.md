# RAG-Based Chatbot System

## Overview
This project aims to build a Retrieval-Augmented Generation (RAG)-based chatbot system. The chatbot utilizes context-aware chunking for efficient document processing and leverages open-source models for embeddings and language generation.

## Features
- **Context-Aware Chunking**: Implements an optimal, manual chunking strategy that uses special chunk markers to separate chunks within documents. This allows for easy extraction of chunks using Python's split() method. Make sure the documents users upload have been chunked using a special chunk marker separator for effective processing.
  
- **Data Processing Tool**: - **Data Processing Tool**: Converts tables in documents into HTML tables to handle challenges such as long tables and merged cells. The tool requires users to bold headers in tables for clarity and automatically identifies table types, including those with 1 header or more than 2 headers.

- **Open-Source Models**: Uses open-source models instead of proprietary ones like those from OpenAI, providing a cost-effective and flexible solution.
  - **Embedding Model**: Utilizes `intfloat/multilingual-e5-small`, which is highly efficient and particularly effective for Vietnamese text.
  - **Language Model**: Uses `Viet-Mistral/Vistral-7B-Chat`, a language model based on Mistral, with continued pretraining on Vietnamese for better generation performance.

## Installation
1. Clone the repository:
```sh
git clone https://github.com/quoctata2911/RAG-based-ChatBot-System.git
```

2. Navigate to the project directory:
```sh
cd RAG-Based-Chatbot-System
```

3. Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Upload your Word .docx documents into the data folder. Ensure that each document has been chunked using a special chunk marker separator as specified in the config.yaml file.

1. Configure the chunk marker:
- Open the `config.yaml` file located in the project directory.
- Locate the parameter defining the chunk marker and adjust it as needed for your document segmentation requirements.

2. Prepare the data:
```sh
python prepare_data.py
```
3. Run the chatbot:
```sh
python chat.py
```

## Project Structure
- **prepare_data.py**: Script to preprocess and chunk documents, converting tables into HTML and segmenting them with chunk markers.
- **chat.py**: Main script to run the chatbot system.

## Models
- **Embedding Model**: We use the `intfloat/multilingual-e5-small` model for generating embeddings. This model is particularly effective for Vietnamese text, outperforming other models in our benchmarks.

- **Language Model**: The language model used is Vistral, a variant of the Mistral model that has been further pre-trained on Vietnamese text for improved performance in language generation tasks.

## Benchmarking and Performance
Through extensive benchmarking, the `intfloat/multilingual-e5-small` model has proven to be the best choice for Vietnamese embeddings, offering a balance of efficiency and performance. The Vistral model enhances language generation capabilities, ensuring the chatbot responds accurately and naturally in Vietnamese.

## Contributions
We welcome contributions to improve the RAG-ChatBot. Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or suggestions, please contact me at [Your Email].
