import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

search_tool = SerperDevTool()

def create_research_agent(use_gpt=True):
    if use_gpt:
        llm = ChatOpenAI(model="gpt-4o-mini")
    else:
        llm = Ollama(model="llama3.1") 

    return Agent(
        role='Research Specialist',
        goal='Conduct thorough research on given topics',
        backstory='You are an experienced researcher with expertise in finding and synthesizing information from various sources.',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )

def create_research_task(agent, topic):
    return Task(
        description=f"Research the following topic and provide a comprehensive summary: {topic}",
        agent=agent,
        expected_output="A detailed summary of the research findings, including key points, trends, and insights related to the topic."
    )

def run_research(topic, use_gpt=True):
    agent = create_research_agent(use_gpt)
    task = create_research_task(agent, topic)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    print("Welcome to the Research Agent!")
    use_gpt = input("Do you want to use GPT? (yes/no): ").lower() == 'yes'
    topic = input("Enter the research topic: ")
    
    result = run_research(topic, use_gpt)
    print("\nResearch Result:")
    print(result)