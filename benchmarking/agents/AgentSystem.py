import json
from typing import Dict, Optional

class Command:
    """Represents a command an agent can issue to a neighboring agent."""
    def __init__(self, name: str, target_agent: str, description: str):
        self.name = name
        self.target_agent = target_agent
        self.description = description

    def __repr__(self) -> str:
        return (f"Command(name='{self.name}', target='{self.target_agent}', "
                f"desc='{self.description[:30]}...')")

class Agent:
    """Represents a single agent in the system."""
    def __init__(self, name: str, prompt: str, commands: Dict[str, Command]):
        self.name = name
        self.prompt = prompt
        self.commands = commands

    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', commands={list(self.commands.keys())})"

    def get_full_prompt(self) -> str:
        """Constructs the full prompt including command descriptions for the LLM."""
        full_prompt = self.prompt
        if self.commands:
            full_prompt += "\n\nYou can use the following commands to delegate tasks:"
            for name, command in self.commands.items():
                full_prompt += f"\n- Command: `{name}`"
                full_prompt += f"\n  - Description: {command.description}"
                full_prompt += f"\n  - Target Agent: {command.target_agent}"
            full_prompt += "YOU MUST USE THESE EXACT COMMANDS TO DELEGATE TASKS. NO OTHER FORMATTING OR COMMANDS ARE ALLOWED."
        return full_prompt

class AgentSystem:
    """
    Loads and holds the entire agent system configuration from a JSON file,
    representing the network of agents and their communication channels.
    """
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents

    @classmethod
    def load_from_json(cls, file_path: str) -> 'AgentSystem':
        """Parses the JSON blueprint and builds the AgentSystem data structure."""
        print(f"Loading agent system from: {file_path}")
        with open(file_path, 'r') as f:
            config = json.load(f)

        agents: Dict[str, Agent] = {}
        for agent_name, agent_data in config.get('agents', {}).items():
            commands: Dict[str, Command] = {}
            for cmd_name, cmd_data in agent_data.get('neighbors', {}).items():
                command = Command(
                    name=cmd_name,
                    target_agent=cmd_data['target_agent'],
                    description=cmd_data['description']
                )
                commands[cmd_name] = command
            
            agent = Agent(
                name=agent_name,
                prompt=agent_data['prompt'],
                commands=commands
            )
            agents[agent_name] = agent
        
        print("Agent system loaded successfully.")
        return cls(agents)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Retrieves an agent by its unique name."""
        return self.agents.get(name)
    
    def get_all_agents(self) -> Dict[str, Agent]:
        """Returns a dictionary of all agents in the system."""
        return self.agents

    def get_insturctions(self) -> str:
        """Generates a summary of the system's instructions for the LLM."""
        instructions = "You are part of a multi-agent system with the following agents:\n"
        for agent in self.agents.values():
            instructions += f"\n- Agent: {agent.name}\n  Prompt: {agent.prompt}\n"
            if agent.commands:
                instructions += "  Commands:\n"
                for cmd in agent.commands.values():
                    instructions += f"    - {cmd.name}: {cmd.description} (target: {cmd.target_agent})\n"
        return instructions

    def __repr__(self) -> str:
        return f"AgentSystem(agents={list(self.agents.keys())})"