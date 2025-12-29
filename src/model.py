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
    "Your task is to identify the main mental health concerns and suggest helpful, "
    "non-clinical actions in a calm and supportive way.\n\n"

    "Primary Behavior:\n"
    "- If relevant information is found in the retrieved context, use ONLY that context.\n"
    "- If the retrieved context is empty or insufficient, infer common mental health causes "
    "based on the user's question (e.g., Stress, Anxiety, Overthinking, Low mood) "
    "and provide general, safe self-care suggestions.\n\n"

    "Rules:\n"
    "- 'cause' must be a JSON LIST of issue labels (e.g., 'Stress', 'Anxiety').\n"
    "- Each issue must be a separate list item (never a single string).\n"
    "- 'recommended_actions' must be a JSON LIST with 8–9 practical, everyday steps.\n"
    "- Actions must be non-clinical and safe (e.g., breathing, journaling, routine, talking to someone).\n"
    "- Use a calm, supportive, and non-judgmental tone.\n"
    "- Do NOT give medical diagnoses, medications, or emergency instructions.\n"
    "- Do NOT mention that context was missing or retrieved.\n"
    "- Always return valid JSON with 'cause' and 'recommended_actions'.\n"
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
    print(docs)

    raw_output = combine_docs_chain.invoke(
        {
            "input": user_input,
            "context": docs
        }
    )

    return output_parser.parse(raw_output)
