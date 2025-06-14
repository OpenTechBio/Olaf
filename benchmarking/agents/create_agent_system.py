import json
import os
from typing import Dict, Any

# A simple class to hold ANSI color codes for terminal output
class Colors:
    """A class to hold ANSI color codes for terminal output."""
    HEADER = '\033[95m'      # Magenta
    OKBLUE = '\033[94m'      # Blue
    OKCYAN = '\033[96m'      # Cyan
    OKGREEN = '\033[92m'     # Green
    WARNING = '\033[93m'     # Yellow
    FAIL = '\033[91m'        # Red
    ENDC = '\033[0m'         # Reset to default
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_output_directory() -> str:
    """Asks the user for an output directory, with a default option."""
    default_dir = "benchmarking/agent_systems"
    dir_prompt = f"{Colors.WARNING}Enter the output directory (press Enter to use '{default_dir}'): {Colors.ENDC}"
    user_input = input(dir_prompt).strip()
    return user_input or default_dir

def define_agents() -> Dict[str, Dict[str, Any]]:
    """Guides the user through defining all agents and their prompts."""
    agents = {}
    print(f"\n{Colors.OKBLUE}--- Agent Definition ---{Colors.ENDC}")
    print("Let's define your agents. Type 'done' when you have no more agents to add.")

    while True:
        prompt_text = f"\n{Colors.WARNING}Enter a unique name for the agent (e.g., 'master_agent') or 'done': {Colors.ENDC}"
        agent_name = input(prompt_text).strip()
        
        if agent_name.lower() == 'done':
            if not agents:
                print(f"{Colors.FAIL}No agents defined. Exiting.{Colors.ENDC}")
                return {}
            break
        
        if not agent_name:
            print(f"{Colors.FAIL}Agent name cannot be empty. Please try again.{Colors.ENDC}")
            continue
            
        if agent_name in agents:
            print(f"{Colors.FAIL}Agent '{agent_name}' already exists. Please use a unique name.{Colors.ENDC}")
            continue

        prompt = input(f"{Colors.WARNING}Enter the system prompt for '{Colors.OKCYAN}{agent_name}{Colors.WARNING}': {Colors.ENDC}").strip()
        agents[agent_name] = {"prompt": prompt, "neighbors": {}}
        print(f"{Colors.OKGREEN}Agent '{Colors.OKCYAN}{agent_name}{Colors.OKGREEN}' added successfully.{Colors.ENDC}")
        
    print(f"\n{Colors.OKBLUE}--- All Agents Defined ---{Colors.ENDC}")
    for name in agents:
        print(f"- {Colors.OKCYAN}{name}{Colors.ENDC}")
    return agents

def connect_agents(agents: Dict[str, Dict[str, Any]]) -> None:
    """Guides the user through connecting agents to each other."""
    print(f"\n{Colors.OKBLUE}--- Agent Connection ---{Colors.ENDC}")
    print("Now, let's define the connections (neighbors) between agents.")
    print("Type 'done' at any point to finish connecting agents.")

    agent_names = list(agents.keys())
    if len(agent_names) < 2:
        print("You need at least two agents to create a connection. Skipping this step.")
        return

    while True:
        print(f"\n{Colors.BOLD}Select the agent that will delegate the task (source agent).{Colors.ENDC}")
        for i, name in enumerate(agent_names):
            print(f"  {i + 1}: {Colors.OKCYAN}{name}{Colors.ENDC}")
        
        source_choice_input = input(f"{Colors.WARNING}Enter the number of the source agent (or 'done'): {Colors.ENDC}").strip()
        if source_choice_input.lower() == 'done':
            break

        try:
            source_idx = int(source_choice_input) - 1
            if not 0 <= source_idx < len(agent_names):
                raise ValueError
            source_agent_name = agent_names[source_idx]
        except (ValueError, IndexError):
            print(f"{Colors.FAIL}Invalid selection. Please enter a number from the list.{Colors.ENDC}")
            continue

        print(f"\nSelected source agent: '{Colors.OKCYAN}{source_agent_name}{Colors.ENDC}'")
        print(f"{Colors.BOLD}Select the agent to delegate to (target agent).{Colors.ENDC}")
        
        # Create a list of valid target choices to check against
        valid_targets = []
        for i, name in enumerate(agent_names):
            if name != source_agent_name:
                print(f"  {i + 1}: {Colors.OKCYAN}{name}{Colors.ENDC}")
                valid_targets.append(name)

        target_choice_input = input(f"{Colors.WARNING}Enter the number of the target agent: {Colors.ENDC}").strip()
        try:
            target_idx = int(target_choice_input) - 1
            # Adjust index for display vs. actual list of agents
            potential_target_name = agent_names[target_idx]
            if potential_target_name not in valid_targets:
                 raise ValueError
            target_agent_name = potential_target_name
        except (ValueError, IndexError):
            print(f"{Colors.FAIL}Invalid selection. Please enter a valid number for a different agent.{Colors.ENDC}")
            continue

        delegation_command = input(f"{Colors.WARNING}Enter the delegation command name (e.g., 'delegate_to_coder'): {Colors.ENDC}").strip()
        description = input(f"{Colors.WARNING}Enter the description for this delegation to '{Colors.OKCYAN}{target_agent_name}{Colors.WARNING}': {Colors.ENDC}").strip()

        # Add the neighbor connection to the source agent
        agents[source_agent_name]["neighbors"][delegation_command] = {
            "target_agent": target_agent_name,
            "description": description
        }
        print(f"{Colors.OKGREEN}Successfully connected '{Colors.OKCYAN}{source_agent_name}{Colors.OKGREEN}' to '{Colors.OKCYAN}{target_agent_name}{Colors.OKGREEN}' via '{delegation_command}'.{Colors.ENDC}")


def save_configuration(agents_config: Dict[str, Any], output_dir: str) -> None:
    """Saves the final configuration to a JSON file."""
    if not agents_config:
        return 

    final_structure = {"agents": agents_config}
    
    os.makedirs(output_dir, exist_ok=True)

    filename_prompt = f"\n{Colors.WARNING}Enter a filename for your agent system (e.g., 'my_research_team.json'): {Colors.ENDC}"
    filename = input(filename_prompt).strip()
    if not filename.endswith('.json'):
        filename += '.json'
        
    file_path = os.path.join(output_dir, filename)

    try:
        with open(file_path, 'w') as f:
            json.dump(final_structure, f, indent=2)
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}Success! Agent configuration saved to: {file_path}{Colors.ENDC}")
    except IOError as e:
        print(f"\n{Colors.FAIL}Error: Could not save the file. {e}{Colors.ENDC}")


def main():
    """Main function to run the interactive agent builder."""
    print(f"{Colors.HEADER}{Colors.BOLD}--- Welcome to the Interactive Agent Configuration Builder ---{Colors.ENDC}")
    output_directory = get_output_directory()
    
    agents_data = define_agents()
    
    if agents_data:
        connect_agents(agents_data)
        save_configuration(agents_data, output_directory)

if __name__ == "__main__":
    main()
