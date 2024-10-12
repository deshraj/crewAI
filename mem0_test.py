from mem0 import MemoryClient

# Step 1: Record preferences based on the past conversation or user input
client = MemoryClient()
messages = [
    {
        "role": "user",
        "content": "Hi there! I'm planning a vacation and could use some advice.",
    },
    {
        "role": "assistant",
        "content": "Hello! I'd be happy to help with your vacation planning. What kind of destination do you prefer?",
    },
    {"role": "user", "content": "I am more of a beach person than a mountain person."},
    {
        "role": "assistant",
        "content": "That's interesting. Do you like hotels or airbnb?",
    },
    {"role": "user", "content": "I like airbnb more."},
]
client.add(messages, user_id="crew_user_1")


# Step 2: Use the preferences to generate a personalized itinerary in CrewAI

from crewai import Agent, Task, Crew, Process  # noqa
from crewai_tools import SerperDevTool  # noqa


search_tool = SerperDevTool()

travel_agent = Agent(
    role="Personal Travel Planner Agent",
    goal="Plan a 5-day trip to San Francisco",
    backstory="""You are a seasoned travel planner, known for your meticulous attention to detail.""",
    verbose=False,
    allow_delegation=False,
    memory=True,
    tools=[search_tool],
)

task1 = Task(
    description="""Find places to live, eat, and visit in San Francisco.""",
    expected_output="A detailed list of places to live, eat, and visit in San Francisco.",
    agent=travel_agent,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[travel_agent],
    tasks=[task1],
    verbose=False,
    process=Process.sequential,
    memory=True,
    memory_config={
        "provider": "mem0",
        "config": {"user_id": "crew_user_1"},
    },
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
