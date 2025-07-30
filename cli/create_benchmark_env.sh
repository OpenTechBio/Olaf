#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the path for the .env file in the script's directory
ENV_FILE_PATH="${SCRIPT_DIR}/.env"

echo "This script will create a .env file to store your OpenAI API key."
echo "The file will be saved in the script's directory: ${SCRIPT_DIR}"
echo "" # Add a blank line for spacing

# Prompt the user for their OpenAI API key
# -p: Display the prompt string
# -s: Silent mode (do not echo input characters) - recommended for keys/passwords
# -r: Raw mode (backslashes are not treated as escape characters)
read -p "Please enter your OpenAI API key: " -s -r OPENAI_API_KEY
echo "" # Add a newline after the hidden input

# Check if the key was entered
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: No API key entered. Exiting."
  exit 1
fi

# Write the key to the .env file in the format OPENAI_KEY:key_value
# Overwrites the file if it already exists
echo "OPENAI_API_KEY=${OPENAI_API_KEY}" > "${ENV_FILE_PATH}"

# Check if the file was created successfully
if [ $? -eq 0 ]; then
  echo "" # Add a blank line
  echo "Successfully saved the OpenAI API key to ${ENV_FILE_PATH}"
  # Optionally, set permissions to be readable only by the user
  chmod 600 "${ENV_FILE_PATH}"
  echo "Set permissions for ${ENV_FILE_PATH} to read-only for the current user (600)."
else
  echo "Error: Failed to write to ${ENV_FILE_PATH}. Please check permissions."
  exit 1
fi

exit 0
