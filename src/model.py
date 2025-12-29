from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings

load_dotenv()

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(index_name="test",embedding=embeddings)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.4)

response_schemas = [
    ResponseSchema(name="cause",description="List of likely causes of the user's mental health concern"),
    ResponseSchema(name="recommended_actions",description="List of non-clinical coping or self-help actions"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

system_prompt = (
    "You are a friendly mental health advisory assistant. "
    "Use ONLY the retrieved context to identify the main mental health issues "
    "and suggest helpful, non-clinical actions.\n\n"

    "Rules:\n"
    "- 'cause' must be a JSON LIST of issue labels (e.g., 'Stress', 'Anxiety').\n"
    "- Each issue must be a separate list item (never a single string).\n"
    "- 'recommended_actions' must be a JSON LIST with 8-9 practical steps.\n"
    "- Use a calm, supportive tone.\n"
    "- Do NOT give medical diagnoses, medications, or emergency advice.\n"
    "- If the context is unclear, return empty lists.\n"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
            + "\n\nUse the following context to answer the user.\n"
            + "Follow the format instructions strictly.\n\n"
            + "Context:\n{context}\n\n"
            + "{format_instructions}"
        ),
        (
            "human",
            "User concern: {input}\n"
            "Task: Identify causes and recommended actions based on the context."
        ),
    ]
)

prompt = prompt.partial(format_instructions=format_instructions)

combine_docs_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)


def mental_health_rag(user_input: str) -> dict:
    retrieval_query = f"""
    Mental health issue and remedies related to: {user_input}.
    Focus on coping strategies, self-help steps, grounding techniques,
    lifestyle changes, and supportive actions.
    """

    docs = retriever.invoke(retrieval_query)

    raw_output = combine_docs_chain.invoke(
        {
            "input": user_input,
            "context": docs
        }
    )

    return output_parser.parse(raw_output)
