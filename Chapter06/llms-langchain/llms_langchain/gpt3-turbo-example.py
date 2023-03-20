import os
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

# Initiatilise the chat model object
openai_key = os.getenv('''OPENAI_API_KEY''')
gpt = ChatOpenAI(model_name='''gpt-3.5-turbo''')

# Create a basic template
template = """Question: {question}

Answer: """

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "Where does Andrew McMahon, author of 'Machine Learning Engineering with Python' work?"

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=gpt
)

# ask the user question
print(llm_chain.run(question))


questions = [
    {'question': "Where does Andrew McMahon, author of 'Machine Learning Engineering with Python' work?"},
    {'question': "What is MLOps?"},
    {'question': "What is ML engineering?"},
    {'question': "What's your favourite flavour of ice cream?"}
]
print(llm_chain.generate(questions))
