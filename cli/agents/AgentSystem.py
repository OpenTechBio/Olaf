import json
from typing import Dict, Optional
from pathlib import Path

CODE_SAMPLES_DIR = Path("cli/code_samples")


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
    # Updated to accept a dictionary of loaded code samples
    def __init__(self, name: str, prompt: str, commands: Dict[str, Command], code_samples: Dict[str, str]):
        self.name = name
        self.prompt = prompt
        self.commands = commands
        self.code_samples = code_samples

    def __repr__(self) -> str:
        # Updated to show if code samples are loaded
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
        Parses the JSON blueprint, reads code sample files from disk,
        and builds the AgentSystem data structure.
        """
        print(f"Loading agent system from: {file_path}")
        blueprint_path = Path(file_path).parent
        with open(file_path, 'r') as f:
            config = json.load(f)

        global_policy = config.get('global_policy', '')
        agents: Dict[str, Agent] = {}
        
        for agent_name, agent_data in config.get('agents', {}).items():
            # --- Load Commands (unchanged) ---
            commands: Dict[str, Command] = {}
            for cmd_name, cmd_data in agent_data.get('neighbors', {}).items():
                commands[cmd_name] = Command(
                    name=cmd_name,
                    target_agent=cmd_data['target_agent'],
                    description=cmd_data['description']
                )

            loaded_samples: Dict[str, str] = {}
            # Get the list of filenames from the JSON, e.g., ["load_data.py", "plot.py"]
            sample_filenames = agent_data.get('code_samples', [])
            
            if sample_filenames:
                print(f"  Loading code samples for '{agent_name}'...")
                for filename in sample_filenames:
                    try:
                        # Construct the full path to the sample file
                        sample_path = CODE_SAMPLES_DIR / filename
                        # Read the file content and store it in the dictionary
                        loaded_samples[filename] = sample_path.read_text(encoding="utf-8")
                        print(f"    ✅ Loaded {filename}")
                    except FileNotFoundError:
                        print(f"    ❌ WARNING: Code sample file not found and will be skipped: {sample_path}")
                    except Exception as e:
                        print(f"    ❌ ERROR: Could not read code sample file {sample_path}: {e}")

            # --- Create Agent with loaded samples ---
            agent = Agent(
                name=agent_name,
                prompt=agent_data['prompt'],
                commands=commands,
                code_samples=loaded_samples  # Pass the dictionary of loaded code
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