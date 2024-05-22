# AI Agents with CrewAI

This project demonstrates the creation and utilization of AI agents to research and write an article using the CrewAI framework. The agents are defined to play specific roles: Content Planner, Content Writer, and Editor. They collaborate to generate a well-structured and insightful blog post on a given topic.

## Prerequisites

- Python 3.7 or higher
- An OpenAI API key for accessing `gpt-3.5-turbo`

## Installation

Ensure you have the required libraries installed. You can install them using the following command:

```bash
pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
```

## Project Structure

The project consists of defining agents and tasks to plan, write, and edit a blog post. Here's a brief overview of the key components:

1. **Agents**: Defined with specific roles, goals, and backstories to simulate real-life roles in content creation.
2. **Tasks**: Detailed descriptions and expected outputs for each stage of the content creation process.
3. **Crew**: A collection of agents and tasks that work together to produce the final output.

## Usage

1. **Define Agents**:
    - **Planner**: Plans the content, including an outline, audience analysis, and SEO keywords.
    - **Writer**: Writes the blog post based on the planner's outline.
    - **Editor**: Edits the blog post to ensure it aligns with the brand's voice and follows journalistic best practices.

2. **Define Tasks**:
    - **Plan Task**: Prioritizes trends, identifies the target audience, and develops a content outline.
    - **Write Task**: Crafts a compelling blog post using the content plan.
    - **Edit Task**: Proofreads the blog post for grammatical errors and brand alignment.

3. **Create Crew**:
    - Combine the agents and tasks into a crew that collaborates to complete the project.

4. **Run the Crew**:
    - Execute the crew to generate the final blog post.

### Example

```python
from crewai import Agent, Task, Crew
import os
from utils import get_openai_api_key
from IPython.display import Markdown

# Set OpenAI API key
openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Define Agents
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article about the topic: {topic}. You collect information that helps the audience learn something and make informed decisions. Your work is the basis for the Content Writer to write an article on this topic.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory="You're working on writing a new opinion piece about the topic: {topic}. You base your writing on the work of the Content Planner, who provides an outline and relevant context about the topic. You follow the main objectives and direction of the outline, as provided by the Content Planner. You also provide objective and impartial insights and back them up with information provided by the Content Planner. You acknowledge in your opinion piece when your statements are opinions as opposed to objective statements.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory="You are an editor who receives a blog post from the Content Writer. Your goal is to review the blog post to ensure that it follows journalistic best practices, provides balanced viewpoints when providing opinions or assertions, and also avoids major controversial topics or opinions when possible.",
    allow_delegation=False,
    verbose=True
)

# Define Tasks
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}."
        "2. Identify the target audience, considering their interests and pain points."
        "3. Develop a detailed content outline including an introduction, key points, and a call to action."
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling blog post on {topic}."
        "2. Incorporate SEO keywords naturally."
        "3. Sections/Subtitles are properly named in an engaging manner."
        "4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion."
        "5. Proofread for grammatical errors and alignment with the brand's voice."
    ),
    expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for grammatical errors and alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
    agent=editor
)

# Create Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)

# Run the Crew
result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})

# Display the result
Markdown(result)
```

## Customization

You can try the process with different topics by modifying the `inputs` parameter:

```python
topic = "YOUR TOPIC HERE"
result = crew.kickoff(inputs={"topic": topic})
Markdown(result)
```

## Additional Resources

For more information on connecting different language models to CrewAI, refer to the [CrewAI documentation](https://docs.crewai.com/how-to/LLM-Connections/).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
