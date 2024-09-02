import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

search_tool = SerperDevTool()

def create_llm(use_gpt=True):
    if use_gpt:
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        return Ollama(model="llama3.1")
    
def create_agents(brand_name, llm):
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources",
        backstory="You are an expert researcher with a knack for finding relevant information quickly.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15  # Increased max iterations
    )

    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        backstory="You are an experienced social media analyst with keen eyes for trends and mentions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15  # Increased max iterations
    )

    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        backstory="You are an expert in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15  # Increased max iterations
    )

    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}",
        backstory="You are a skilled data analyst and report writer, adept at presenting insights clearly.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15  # Increased max iterations
    )

    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

def create_tasks(brand_name, agents):
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of their online presence, key information, and recent activities.",
        agent=agents[0],
        expected_output="A structured summary containing: \n1. Brief overview of {brand_name}\n2. Key online platforms and follower counts\n3. Recent notable activities or campaigns\n4. Main products or services\n5. Any recent news or controversies"
    )

    monitoring_task = Task(
        description=f"Monitor social media platforms for mentions of '{brand_name}' in the last 24 hours. Provide a summary of the mentions.",
        agent=agents[1],
        expected_output="A structured report containing: \n1. Total number of mentions\n2. Breakdown by platform (e.g., Twitter, Instagram, Facebook)\n3. Top 5 most engaging posts or mentions\n4. Any trending hashtags associated with {brand_name}\n5. Notable influencers or accounts mentioning {brand_name}"
    )

    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}. Categorize them as positive, negative, or neutral.",
        agent=agents[2],
        expected_output="A sentiment analysis report containing: \n1. Overall sentiment distribution (% positive, negative, neutral)\n2. Key positive themes or comments\n3. Key negative themes or comments\n4. Any notable changes in sentiment compared to previous periods\n5. Suggestions for sentiment improvement if necessary"
    )

    report_generation_task = Task(
        description=f"Generate a comprehensive report about {brand_name} based on the research, social media mentions, and sentiment analysis. Include key insights and recommendations.",
        agent=agents[3],
        expected_output="A comprehensive report structured as follows: \n1. Executive Summary\n2. Brand Overview\n3. Social Media Presence Analysis\n4. Sentiment Analysis\n5. Key Insights\n6. Recommendations for Improvement\n7. Conclusion"
    )

    return [research_task, monitoring_task, sentiment_analysis_task, report_generation_task]

def run_social_media_monitoring(brand_name, use_gpt=True, max_retries=3):
    llm = create_llm(use_gpt)
    agents = create_agents(brand_name, llm)
    tasks = create_tasks(brand_name, agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )

    for attempt in range(max_retries):
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print("Max retries reached. Unable to complete the task.")
                return None

if __name__ == "__main__":
    print("Welcome to the Social Media Monitoring Crew!")
    use_gpt = input("Do you want to use GPT? (yes/no): ").lower() == 'yes'
    brand_name = input("Enter the name of the brand or influencer you want to research: ")
    
    result = run_social_media_monitoring(brand_name, use_gpt)
    
    if result:
        print("\n", "="*50, "\n")
        print("Final Report:")
        print(result)
    else:
        print("Failed to generate the report. Please try again later.")