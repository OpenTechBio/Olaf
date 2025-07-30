import json
import os
from typing import Dict, Any
from pathlib import Path

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

# Define the directory where code samples are stored
CODE_SAMPLES_DIR = Path("cli/code_samples")

def define_global_policy() -> str:
    """Asks the user to define a global policy for all agents."""
    print(f"\n{Colors.OKBLUE}--- Global Policy Definition ---{Colors.ENDC}")
    print("First, let's define a global policy. This is a set of general guidelines that all agents should follow.")
    policy_prompt = f"{Colors.WARNING}Enter the global policy text (e.g., 'Always be concise and professional'): {Colors.ENDC}"
    policy = input(policy_prompt).strip()
    if not policy:
        print(f"{Colors.OKCYAN}No global policy provided. Proceeding without one.{Colors.ENDC}")
        return ""
    print(f"{Colors.OKGREEN}Global policy set successfully.{Colors.ENDC}")
    return policy

def get_output_directory() -> str:
    """Asks the user for an output directory, with a default option."""
    default_dir = "cli/agent_systems"
    dir_prompt = f"{Colors.WARNING}Enter the output directory (press Enter to use '{default_dir}'): {Colors.ENDC}"
    user_input = input(dir_prompt).strip()
    return user_input or default_dir

def define_agents() -> Dict[str, Dict[str, Any]]:
    """Guides the user through defining all agents and their prompts."""
    agents = {}
    print(f"\n{Colors.OKBLUE}--- Agent Definition ---{Colors.ENDC}")
    print("Now, let's define your agents. Type 'done' when you have no more agents to add.")

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
        # Initialize agent with an empty list for code samples
        agents[agent_name] = {"prompt": prompt, "neighbors": {}, "code_samples": []}
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
        print(f"{Colors.WARNING}You need at least two agents to create a connection. Skipping this step.{Colors.ENDC}")
        return

    while True:
        print(f"\n{Colors.BOLD}Select the agent that will delegate the task (source agent).{Colors.ENDC}")
        for i, name in enumerate(agent_names):
            print(f"  {i + 1}: {Colors.OKCYAN}{name}{Colors.ENDC}")
        source_choice_input = input(f"{Colors.WARNING}Enter the number of the source agent (or 'done'): {Colors.ENDC}").strip()
        if source_choice_input.lower() == 'done': break
        try:
            source_idx = int(source_choice_input) - 1
            if not 0 <= source_idx < len(agent_names): raise ValueError
            source_agent_name = agent_names[source_idx]
        except (ValueError, IndexError):
            print(f"{Colors.FAIL}Invalid selection. Please enter a number from the list.{Colors.ENDC}")
            continue
        print(f"\nSelected source agent: '{Colors.OKCYAN}{source_agent_name}{Colors.ENDC}'")
        print(f"{Colors.BOLD}Select the agent to delegate to (target agent).{Colors.ENDC}")
        valid_targets = [name for name in agent_names if name != source_agent_name]
        for i, name in enumerate(valid_targets):
            print(f"  {i + 1}: {Colors.OKCYAN}{name}{Colors.ENDC}")
        target_choice_input = input(f"{Colors.WARNING}Enter the number of the target agent: {Colors.ENDC}").strip()
        try:
            target_idx = int(target_choice_input) - 1
            if not 0 <= target_idx < len(valid_targets): raise ValueError
            target_agent_name = valid_targets[target_idx]
        except (ValueError, IndexError):
            print(f"{Colors.FAIL}Invalid selection. Please enter a valid number.{Colors.ENDC}")
            continue
        delegation_command = input(f"{Colors.WARNING}Enter the delegation command name (e.g., 'delegate_to_coder'): {Colors.ENDC}").strip()
        description = input(f"{Colors.WARNING}Enter the description for this delegation to '{Colors.OKCYAN}{target_agent_name}{Colors.WARNING}': {Colors.ENDC}").strip()
        agents[source_agent_name]["neighbors"][delegation_command] = {
            "target_agent": target_agent_name,
            "description": description
        }
        print(f"{Colors.OKGREEN}Successfully connected '{Colors.OKCYAN}{source_agent_name}{Colors.OKGREEN}' to '{Colors.OKCYAN}{target_agent_name}{Colors.OKGREEN}' via '{delegation_command}'.{Colors.ENDC}")

def assign_code_samples(agents: Dict[str, Dict[str, Any]]) -> None:
    """Interactively assign code sample files to agents."""
    print(f"\n{Colors.OKBLUE}--- Code Sample Assignment ---{Colors.ENDC}")
    
    # Ensure the code samples directory exists
    CODE_SAMPLES_DIR.mkdir(exist_ok=True, parents=True)
    
    try:
        sample_files = [f.name for f in CODE_SAMPLES_DIR.glob("*.py")]
    except Exception as e:
        print(f"{Colors.FAIL}Could not read code samples directory: {e}{Colors.ENDC}")
        return

    if not sample_files:
        print(f"{Colors.WARNING}No code samples found in '{CODE_SAMPLES_DIR}'. Skipping assignment.{Colors.ENDC}")
        print(f"You can add `.py` files to this directory to make them available.")
        return

    for agent_name, agent_data in agents.items():
        while True:
            assign_prompt = f"\n{Colors.WARNING}Assign code samples to agent '{Colors.OKCYAN}{agent_name}{Colors.WARNING}'? (y/n): {Colors.ENDC}"
            if input(assign_prompt).strip().lower() != 'y':
                break

            print(f"{Colors.BOLD}Available code samples:{Colors.ENDC}")
            for i, filename in enumerate(sample_files):
                print(f"  {i + 1}: {Colors.OKCYAN}{filename}{Colors.ENDC}")

            choice_prompt = f"{Colors.WARNING}Enter a number to add a sample, or type 'done': {Colors.ENDC}"
            choice = input(choice_prompt).strip().lower()

            if choice == 'done':
                break
            
            try:
                index = int(choice) - 1
                if not 0 <= index < len(sample_files):
                    raise ValueError
                
                chosen_file = sample_files[index]
                if chosen_file not in agent_data["code_samples"]:
                    agent_data["code_samples"].append(chosen_file)
                    print(f"{Colors.OKGREEN}Assigned '{chosen_file}' to '{agent_name}'.{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}'{chosen_file}' is already assigned to this agent.{Colors.ENDC}")

            except (ValueError, IndexError):
                print(f"{Colors.FAIL}Invalid selection. Please enter a valid number.{Colors.ENDC}")

def save_configuration(global_policy: str, agents_config: Dict[str, Any], output_dir: str) -> None:
    """Saves the final configuration, including the global policy, to a JSON file."""
    if not agents_config:
        return 

    final_structure = {
        "global_policy": global_policy,
        "agents": agents_config
    }
    
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

    global_policy_text = define_global_policy()
    output_directory = get_output_directory()
    agents_data = define_agents()
    
    if agents_data:
        connect_agents(agents_data)
        assign_code_samples(agents_data)
        save_configuration(global_policy_text, agents_data, output_directory)

if __name__ == "__main__":
    main()