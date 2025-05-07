#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE_PATH="${SCRIPT_DIR}/.env"

echo "This script will create a .env file to store your API keys (OpenAI, Claude, Gemini)."
echo "The file will be saved in: ${ENV_FILE_PATH}"
echo ""

# Prompt for API keys (optional, but must provide at least one)
read -p "Please enter your OpenAI API key (optional): " -s -r OPENAI_API_KEY
echo ""
read -p "Please enter your Claude API key (optional): " -s -r CLAUDE_API_KEY
echo ""
read -p "Please enter your Gemini API key (optional): " -s -r GEMINI_API_KEY
echo ""

# Check if at least one was entered
if [ -z "$OPENAI_API_KEY" ] && [ -z "$CLAUDE_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
  echo "⚠️  Error: You must enter at least one API key. Exiting."
  exit 1
fi

# Write each non-empty key to the .env file
echo "# Auto-generated .env file with API keys" > "${ENV_FILE_PATH}"

[ -n "$OPENAI_API_KEY" ] && echo "OPENAI_API_KEY=${OPENAI_API_KEY}" >> "${ENV_FILE_PATH}"
[ -n "$CLAUDE_API_KEY" ] && echo "CLAUDE_API_KEY=${CLAUDE_API_KEY}" >> "${ENV_FILE_PATH}"
[ -n "$GEMINI_API_KEY" ] && echo "GEMINI_API_KEY=${GEMINI_API_KEY}" >> "${ENV_FILE_PATH}"

# Set secure permissions
chmod 600 "${ENV_FILE_PATH}"

echo ""
echo "✅ Successfully saved keys to ${ENV_FILE_PATH}"
echo "🔒 Permissions set to read/write for user only (600)."

exit 0