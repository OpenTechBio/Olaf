import json
from typing import Dict, Optional
from pathlib import Path

# Import the central OLAF_HOME path from our config module
from olaf.config import OLAF_HOME

# 1. The user-specific directory (for custom samples)
USER_CODE_SAMPLES_DIR = OLAF_HOME / "code_samples"
USER_CODE_SAMPLES_DIR.mkdir(exist_ok=True) # Ensure it exists

# 2. The package-internal directory (for default samples), found relative to this file
PACKAGE_CODE_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "code_samples"


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
    def __init__(self, name: str, prompt: str, commands: Dict[str, Command], code_samples: Dict[str, str]):
        self.name = name
        self.prompt = prompt
        self.commands = commands
        self.code_samples = code_samples

    def __repr__(self) -> str:
        sample_keys = list(self.code_samples.keys())
        return f"Agent(name='{self.name}', commands={list(self.commands.keys())}, samples={sample_keys})"

    def get_full_prompt(self, global_policy=None) -> str:
        """Constructs the full prompt including the global policy and command descriptions."""
        full_prompt = ""
        if global_policy:
            full_prompt += f"**GLOBAL POLICY**: {global_policy}\n\n---\n\n"
        
        full_prompt += self.prompt

        if self.commands:
            full_prompt += "\n\nYou can use the following commands to delegate tasks:"
            for name, command in self.commands.items():
                full_prompt += f"\n- Command: `{name}`"
                full_prompt += f"\n  - Description: {command.description}"
                full_prompt += f"\n  - Target Agent: {command.target_agent}"
            full_prompt += "\n\n**YOU MUST USE THESE EXACT COMMANDS TO DELEGATE TASKS. NO OTHER FORMATTING OR COMMANDS ARE ALLOWED.**"
        
        if self.code_samples:
            full_prompt += "\n  - Code Samples Available:"
            for sample_name in self.code_samples.keys():
                full_prompt += f"\n    - `{sample_name}`"
  
        return full_prompt


class AgentSystem:
    """
    Loads and holds the entire agent system configuration from a JSON file,
    including the global policy and the network of agents.
    """
    def __init__(self, global_policy: str, agents: Dict[str, Agent]):
        self.global_policy = global_policy
        self.agents = agents

    @classmethod
    def load_from_json(cls, file_path: str) -> 'AgentSystem':
        """
        Parses the JSON blueprint, reads code sample files from disk from both user
        and package locations, and builds the AgentSystem data structure.
        """
        print(f"Loading agent system from: {file_path}")
        with open(file_path, 'r') as f:
            config = json.load(f)

        global_policy = config.get('global_policy', '')
        agents: Dict[str, Agent] = {}
        
        for agent_name, agent_data in config.get('agents', {}).items():
            commands: Dict[str, Command] = {}
            for cmd_name, cmd_data in agent_data.get('neighbors', {}).items():
                commands[cmd_name] = Command(
                    name=cmd_name,
                    target_agent=cmd_data['target_agent'],
                    description=cmd_data['description']
                )

            loaded_samples: Dict[str, str] = {}
            sample_filenames = agent_data.get('code_samples', [])
            
            if sample_filenames:
                print(f"  Loading code samples for '{agent_name}'...")
                for filename in sample_filenames:
                    user_path = USER_CODE_SAMPLES_DIR / filename
                    package_path = PACKAGE_CODE_SAMPLES_DIR / filename
                    
                    # Default to package path, but overwrite if user path exists
                    path_to_load = None
                    source_label = ""
                    if user_path.exists():
                        path_to_load = user_path
                        source_label = "User"
                    elif package_path.exists():
                        path_to_load = package_path
                        source_label = "Package"

                    if path_to_load:
                        try:
                            loaded_samples[filename] = path_to_load.read_text(encoding="utf-8")
                            print(f"    ✅ Loaded {filename} (from {source_label})")
                        except Exception as e:
                            print(f"    ❌ ERROR: Could not read code sample file {path_to_load}: {e}")
                    else:
                        print(f"    ❌ WARNING: Code sample file '{filename}' not found in any location.")

            agent = Agent(
                name=agent_name,
                prompt=agent_data['prompt'],
                commands=commands,
                code_samples=loaded_samples
            )
            agents[agent_name] = agent
        
        print("Agent system loaded successfully.")
        return cls(global_policy, agents)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Retrieves an agent by its unique name."""
        return self.agents.get(name)
    
    def get_all_agents(self) -> Dict[str, Agent]:
        """Returns a dictionary of all agents in the system."""
        return self.agents

    def get_instructions(self) -> str:
        """Generates a summary of the system's instructions, including the global policy."""
        instructions = f"**GLOBAL POLICY FOR ALL AGENTS**: {self.global_policy}\n\n---\n\n"
        instructions += "**SYSTEM AGENTS**:\n"
        for agent in self.agents.values():
            instructions += f"\n- **Agent**: {agent.name}\n  - **Prompt**: {agent.prompt}\n"
            if agent.commands:
                instructions += "  - **Commands**:\n"
                for cmd in agent.commands.values():
                    instructions += f"    - `{cmd.name}`: {cmd.description} (delegates to: {cmd.target_agent})\n"
        return instructions

    def __repr__(self) -> str:
        return f"AgentSystem(global_policy='{self.global_policy[:40]}...', agents={list(self.agents.keys())})"