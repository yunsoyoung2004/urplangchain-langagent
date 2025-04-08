
from utils.stage_classifier import classify_stage
from agents import empathy_agent, mi_agent, cbt_agent, ppi_agent, action_agent

def route_user_input(user_input):
    stage = classify_stage(user_input)
    if stage == "precontemplation":
        return mi_agent.run(user_input)
    elif stage == "contemplation":
        return cbt_agent.run(user_input)
    elif stage == "preparation":
        return ppi_agent.run(user_input)
    elif stage == "action":
        return action_agent.run(user_input)
    return empathy_agent.run(user_input)
