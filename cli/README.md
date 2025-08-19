# OLAF CLI: The Open-source Language Agent Framework 🚀

**The OLAF CLI is a powerful command-line interface for building, testing, and running sandboxed, multi-agent AI systems.** 

It provides a robust framework for orchestrating multiple language agents that can collaborate to perform complex tasks, such as data analysis, in a secure and isolated environment.

At its core, OLAF allows you to define a team of specialized AI agents in a simple JSON "blueprint." You can then deploy this team into a secure sandbox (powered by Docker or Singularity) with a specific dataset and give them a high-level task to solve.

## Key Features

  * **Multi-Agent Blueprints:** Define agents, their specialized prompts, and how they delegate tasks to each other using a simple JSON configuration.
  * **Secure Sandboxing:** Execute agent-generated code in an isolated environment using **Docker** or **Singularity** to protect your host system.
  * **Interactive & Automated Modes:** Run agent systems in a turn-by-turn interactive chat for debugging or in a fully automated mode for benchmarking.
  * **Data Curation:** Includes tools to browse and download single-cell datasets from the CZI CELLxGENE Census to easily test your agents.
  * **Configuration Management:** Easily manage API keys and application settings with built-in commands.
  * **User-Friendly CLI:** A guided, interactive experience helps you configure every run, with flags available to override settings for use in scripts.

## Installation

### Prerequisites

Before installing OLAF, you need to have the following installed and configured on your system:

1.  **Python** (version 3.9 or higher)
2.  **Pip** (Python's package installer)
3.  **A Sandbox Backend:**
      * **Docker:** Must be installed and the Docker daemon must be running.
      * **Singularity (Apptainer):** Must be installed on your system.

### Install from PyPI (Recommended)
Coming soon!

### Install from Source (For Developers)

To install the latest development version, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/OpenTechBio/Olaf
cd olaf/cli/olaf
pip install -e .
```

-----

## 🚀 Quick Start Guide

This guide will walk you through setting up your API key, downloading a dataset, and launching your first interactive agent session in just a few steps.

### Step 1: Configure Your API Key

First, tell OLAF about your OpenAI API key. This is a one-time setup.

```bash
olaf config set-openai-key "sk-YourSecretKeyGoesHere"
```

Your key will be stored securely in a local `.env` file within the OLAF configuration directory.

### Step 2: Download a Dataset

Next, let's get some data for our agents to analyze. Run the `datasets` command to browse and download a sample dataset from the CZI CELLxGENE Census.

```bash
olaf datasets
```

Follow the prompts to list versions and datasets, then use the `download` command as instructed.

### Step 3: Run an Agent System\!

Now you're ready to run an agent system. The `run` command is fully interactive if you don't provide any flags. It will guide you through selecting a blueprint, a dataset, and a sandbox environment.

```bash
olaf run interactive
```

This will trigger a series of prompts:

1.  **Select Agent System Blueprint:** Choose one of the default systems (from the Package) or one you've created (from User).
2.  **Select a driver agent:** Choose which agent in the system will receive the first instruction.
3.  **Select Dataset:** Pick the dataset you downloaded in Step 2.
4.  **Choose a sandbox backend:** Select `docker` or `singularity`.
5.  **Choose an LLM backend:** Select `chatgpt` or `ollama`.

After configuration, the session will begin, and you can start giving instructions to your agent team\!

-----

## Command Reference

OLAF's commands are organized into logical groups.

### `olaf run`

The main command for executing an agent system.

  * **Run interactively (recommended for manual use):**
    ```bash
    olaf run interactive
    ```
  * **Run automatically for 5 turns:**
    ```bash
    olaf run auto --turns 5 --prompt "Analyze this dataset and generate a UMAP plot."
    ```
  * **Run with all options specified (for scripting):**
    ```bash
    olaf run interactive \
      --blueprint ~/.local/share/olaf/agent_systems/my_custom_system.json \
      --driver-agent data_analyst \
      --dataset ~/.local/share/olaf/datasets/my_data.h5ad \
      --sandbox docker \
      --llm chatgpt
    ```

### `olaf create-system`

Tools for building new agent system blueprints.

  * **Start the interactive builder:**
    ```bash
    olaf create-system
    ```
  * **Create a minimal blueprint quickly:**
    ```bash
    olaf create-system quick --name my-first-system
    ```

### `olaf datasets`

Tools for managing datasets.

  * **Start the interactive dataset browser:**
    ```bash
    olaf datasets
    ```
  * **Download a specific dataset directly:**
    ```bash
    olaf datasets download --version stable --dataset-id "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    ```

### `olaf config`

Manage your OLAF configuration.

  * **Set your OpenAI API key:**
    ```bash
    olaf config set-openai-key "sk-..."
    ```

-----

## Configuration

OLAF stores all user-generated content and configuration in a central directory. You can override this location by setting the `OLAF_HOME` environment variable.

  * **Default Location:**
      * **Linux:** `~/.local/share/olaf/`
      * **macOS:** `~/Library/Application Support/olaf/`
      * **Windows:** `C:\Users\<user>\AppData\Local\OpenTechBio\olaf\`
  * **Configuration File:** API keys are stored in `$OLAF_HOME/.env`.
  * **Agent Systems:** Custom blueprints are saved to `$OLAF_HOME/agent_systems/`.
  * **Datasets:** Downloaded datasets are stored in `$OLAF_HOME/datasets/`.
  * **Run Outputs:** Code snippets and logs from agent runs are saved to `$OLAF_HOME/runs/`.