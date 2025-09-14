from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-small", api_version="2023-05-15")
vector_store = FAISS.load_local("hk_tourism_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", api_version="2023-05-15")

prompt = PromptTemplate.from_template("""
You are an expert HK travel planner. Use the retrieved HK data to create a personalized itinerary.
User requirements: {query}
Retrieved data: {context}
Output: Day-by-day schedule with attractions/events, MTR/bus transport, costs, and HK-specific tips.
""")

rag_chain = ({"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "query": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# Test
print(rag_chain.invoke("Plan a 3-day HK family trip with $500 budget, focusing on Disneyland and food."))