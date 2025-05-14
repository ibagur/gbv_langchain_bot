from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents import AgentExecutor, Tool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType

import gradio as gr

# Add presets for Gradio theme
from app_modules.presets import * 

import os
# Use environment variable for OpenAI API key
# Make sure to set OPENAI_TOKEN environment variable before running the app
# Example: export OPENAI_TOKEN=your-api-key
# DO NOT hardcode API keys in the source code

# Define the LLM chat model
#model = 'gpt-3.5-turbo'
model = 'gpt-3.5-turbo-16k'
#model = 'gpt-4'
token_limit = 4000 if model == 'gpt-3.5-turbo' else 16000
memory_token_limit = token_limit//2
temperature = 0
llm = ChatOpenAI(temperature=temperature, model=model)

# Load existing vectorstore
persist_dir = "./chroma"
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
vectorstore.persist()

# Create Retrieval Chain with sources
## It returns a dictionary with at least the 'answer' and the 'sources' as metadata if return_source_documents=True
qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="mmr"),
            #retriever=vectorstore.as_retriever(),
            #return_source_documents=True, 
            max_tokens_limit=token_limit
        )

# Define tools
wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(
        name="GBV Q&A Bot System",
        #func=qa,
        func=lambda question: qa({"question": question}, return_only_outputs=True),
        description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
        #return_direct=True, # use the agent as a router and directly return the result
    ),
    Tool(
        name='Wikipedia',
        func=wikipedia.run,
        description='You must only use this tool if you cannot find answers with the other tools. Useful for when you need to look for answers in the Wikipedia.'
    )
]

# Create Conversational Buffer Memory
#memory = ConversationBufferMemory(memory_key="chat_history", input_key='input', output_key="output", return_messages=True)
# Create Conversational Summary Buffer Memory
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", input_key='input', output_key="output", return_messages=True, max_token_limit=memory_token_limit)

# Initialize Re-Act agent and create Agent Executor Chain
react = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, max_iterations=3, early_stopping_method='generate',  memory=memory)

agent_chain = AgentExecutor.from_agent_and_tools(
                agent=react.agent, tools=tools, verbose=True, memory=memory, return_intermediate_steps=True, return_source_documents=False, handle_parsing_errors=True)

# Add custom CSS
with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()
# split sources string in a source per line
def split_sources(sources):
  split_sources = sources.split(",")
  new_sources= "\n".join(["* " + s.strip() for s in split_sources])
  return new_sources

# extract sources when applicable
def get_sources(result):
    if result['intermediate_steps']:
        if result['intermediate_steps'][0][0].tool == "Wikipedia":
            sources = "\n\nSources: Wikipedia"
        elif result['intermediate_steps'][0][0].tool == '_Exception':
            sources = None
        else:
            sources = "\n\nSources:\n" + split_sources(result['intermediate_steps'][0][1]['sources'])
    else:
        sources = None
    return sources
   
with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    
    gr.Markdown(
        """
        # 🦜🔗 Ask the GBV in Emergencies Q&A Bot!
        This generative model has been trained on various sources covering themes on Gender-Based Violence response in Humanitarian Settings. This AI agent might complement the replies with additional information retrieved from Wikipedia sources. You can maintain a natural language conversation with it in order to retrieve information on this area of knowledge.

        Example questions:
        - What are the GBV guiding principles?
        - Which UN agency leads the GBV response in emergencies?
        - How can we engage men and boys in GBV prevention and response? 
        - Please outline a strategy to minimize GBV risks in a temporary settlement
        - What is the integration factor between GBV and SRH?
        """
    )
    
    # Start chatbot with welcome from bot
    chatbot = gr.Chatbot([(None,'How can I help you?')]).style(height=400)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0] # get if from most recent history element
        #bot_message  = conversation.run(user_message)
        #user_message = user_message + " Please provide the source documents" # to alter the prompt and provide sources
        response = agent_chain(user_message)
        sources = get_sources(response)
        bot_message = response['output'] if not sources else response['output'] + sources
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            #time.sleep(0.05)
            yield history

    response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

demo.queue()
demo.launch()