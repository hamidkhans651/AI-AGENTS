from langchain_community.llms import Ollama
from crewai import agent,task,crew,process

model = Ollama(model= "llama3")

email =" pakistani guy sending make an ai agent"


classifier = agent(
    role = "enail classifier",
    goal =" acuractlet classify emails based on their importance.give every email one  of those ratings:importance,casual,or spam",
    verbose = True,
    allow_delegation = False,
    llm = model
)

responder = agent(
    role = "email responder ",
    goal = 'based on the importance of the email,write a consie and simpl',
    backstory = "you are an AI assistant whose onyny job is to respond to eamils accurately and honestly.do not be afradi to ignore emails if they are not importan. your job is to help the user mangae thier inbox",
    verbose = True,
    allow_delagation = False,
    llm = model
   
)
classify_email = task(
    descripton = f"classify the following email: {email} ", 
    agent= classifier,
    expected_output = "one of these three options: 'important','casual', or 'spam'",

)



responder_email = task(
    description = f"responed to the email :{email} based on the importance by the 'classifier' agent",
    agent = responder,
    expected_output = "a very response to email  based on the importance provided by the 'classifier' agent."
)


crew = crew(
    agent [ classifier,responder],
    task = [classify_email,responder_email],
    verbose =2,
    process = process.sequential

)

output = crew.kickoff()
print(output)