"""
UAV Control Agent
An intelligent agent that understands natural language commands and controls drones using the UAV API
Uses LangChain 1.0+ with modern @tool decorator pattern
"""
from langchain_classic.agents import create_react_agent
from langchain_classic.agents import AgentExecutor
from langchain_classic.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from uav_api_client import UAVAPIClient
from uav_langchain_tools import create_uav_tools
from template.agent_prompt import AGENT_PROMPT
from template.parsing_error import PARSING_ERROR_TEMPLATE
from typing import Optional, Dict, Any
import json
import os
from pathlib import Path


def load_llm_settings(settings_path: str = "llm_settings.json") -> Optional[Dict[str, Any]]:
    """Load LLM settings from JSON file"""
    try:
        path = Path(settings_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LLM settings from {settings_path}: {e}")
    return None


def prompt_user_for_llm_config() -> Dict[str, Any]:
    """Prompt user to select LLM provider and model"""
    settings = load_llm_settings()

    if not settings or 'provider_configs' not in settings:
        print("⚠️  No llm_settings.json found or invalid format. Using command line arguments.")
        return {}

    provider_configs = settings['provider_configs']
    selected_provider = settings.get('selected_provider', '')

    print("\n" + "="*60)
    print("🤖 LLM Provider Configuration")
    print("="*60)

    # Show available providers
    providers = list(provider_configs.keys())
    print("\nAvailable providers:")
    for i, provider in enumerate(providers, 1):
        config = provider_configs[provider]
        default_marker = " (selected in settings)" if provider == selected_provider else ""
        print(f"  {i}. {provider}{default_marker}")
        print(f"     Type: {config.get('type', 'unknown')}")
        print(f"     Base URL: {config.get('base_url', 'N/A')}")
        print(f"     Requires API Key: {config.get('requires_api_key', False)}")

    # Prompt for provider selection
    print(f"\nSelect a provider (1-{len(providers)}) [default: {selected_provider or providers[0]}]: ", end='')
    provider_choice = input().strip()

    if not provider_choice:
        # Use default
        if selected_provider and selected_provider in providers:
            chosen_provider = selected_provider
        else:
            chosen_provider = providers[0]
    else:
        try:
            idx = int(provider_choice) - 1
            if 0 <= idx < len(providers):
                chosen_provider = providers[idx]
            else:
                print(f"Invalid choice. Using default: {selected_provider or providers[0]}")
                chosen_provider = selected_provider or providers[0]
        except ValueError:
            print(f"Invalid input. Using default: {selected_provider or providers[0]}")
            chosen_provider = selected_provider or providers[0]

    config = provider_configs[chosen_provider]
    print(f"\n✅ Selected provider: {chosen_provider}")

    # Show available models
    default_models = config.get('default_models', [])
    default_model = config.get('default_model', '')

    if default_models:
        print("\nAvailable models:")
        for i, model in enumerate(default_models, 1):
            default_marker = " (default)" if model == default_model else ""
            print(f"  {i}. {model}{default_marker}")
        print(f"  {len(default_models) + 1}. Custom model (enter manually)")

        print(f"\nSelect a model (1-{len(default_models) + 1}) [default: {default_model}]: ", end='')
        model_choice = input().strip()

        if not model_choice:
            chosen_model = default_model
        else:
            try:
                idx = int(model_choice) - 1
                if 0 <= idx < len(default_models):
                    chosen_model = default_models[idx]
                elif idx == len(default_models):
                    # Custom model
                    print("Enter custom model name: ", end='')
                    chosen_model = input().strip() or default_model
                else:
                    print(f"Invalid choice. Using default: {default_model}")
                    chosen_model = default_model
            except ValueError:
                print(f"Invalid input. Using default: {default_model}")
                chosen_model = default_model
    else:
        # No predefined models, ask for custom input
        print(f"\nEnter model name [default: {default_model}]: ", end='')
        chosen_model = input().strip() or default_model

    print(f"✅ Selected model: {chosen_model}")

    # Determine provider type
    provider_type = config.get('type', 'ollama')
    if provider_type == 'openai-compatible':
        if 'api.openai.com' in config.get('base_url', ''):
            llm_provider = 'openai'
        else:
            llm_provider = 'openai-compatible'
    else:
        llm_provider = provider_type

    # Get API key if required
    api_key = config.get('api_key', '').strip()
    if config.get('requires_api_key', False) and not api_key:
        print("\n⚠️  This provider requires an API key.")
        print("Enter API key (or press Enter to use environment variable): ", end='')
        api_key = input().strip()

    result = {
        'llm_provider': llm_provider,
        'llm_model': chosen_model,
        'llm_base_url': config.get('base_url'),
        'llm_api_key': api_key if api_key else None,
        'provider_name': chosen_provider
    }

    print("\n" + "="*60)
    print("✅ Configuration complete!")
    print("="*60)
    print(f"Provider: {chosen_provider}")
    print(f"Type: {llm_provider}")
    print(f"Model: {chosen_model}")
    print(f"Base URL: {config.get('base_url')}")
    if api_key:
        print(f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '****'}")
    print("="*60 + "\n")

    return result


class UAVControlAgent:
    """Intelligent agent for controlling UAVs using natural language"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        uav_api_key: Optional[str] = None,
        llm_provider: str = "ollama",
        llm_model: str = "llama2",
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        temperature: float = 0.1,
        verbose: bool = True,
        debug: bool = False
    ):
        """
        Initialize the UAV Control Agent

        Args:
            base_url: Base URL of the UAV API server
            uav_api_key: API key for UAV server authentication (None = USER role, or provide SYSTEM/ADMIN key)
            llm_provider: LLM provider ('ollama', 'openai', 'openai-compatible')
            llm_model: Model name (e.g., 'llama2', 'gpt-4o-mini', 'deepseek-chat')
            llm_api_key: API key for LLM provider (required for openai/openai-compatible)
            llm_base_url: Custom base URL for LLM API (for openai-compatible providers)
            temperature: LLM temperature (lower = more deterministic)
            verbose: Enable verbose output for agent reasoning
            debug: Enable debug output for connection and setup info
        """
        self.client = UAVAPIClient(base_url, api_key=uav_api_key)
        self.verbose = verbose
        self.debug = debug

        if self.debug:
            print("\n" + "="*60)
            print("🔧 UAV Agent Initialization - Debug Mode")
            print("="*60)
            print(f"UAV API Server: {base_url}")
            print(f"LLM Provider: {llm_provider}")
            print(f"LLM Model: {llm_model}")
            print(f"Temperature: {temperature}")
            print(f"Verbose: {verbose}")
            print()

        # Test UAV API connection
        if self.debug:
            print("🔌 Testing UAV API connection...")
        try:
            session = self.client.get_current_session()
            if self.debug:
                print(f"✅ Connected to UAV API")
                print(f"   Session: {session.get('name', 'Unknown')}")
                print(f"   Task: {session.get('task', 'Unknown')}")
                print()
        except Exception as e:
            if self.debug:
                print(f"⚠️  Warning: Could not connect to UAV API: {e}")
                print(f"   Make sure the UAV server is running at {base_url}")
                print()

        # Initialize LLM based on provider
        if self.debug:
            print(f"🤖 Initializing LLM provider: {llm_provider}")

        if llm_provider == "ollama":
            if self.debug:
                print(f"   Using Ollama with model: {llm_model}")
                print(f"   Ollama URL: http://localhost:11434 (default)")

            self.llm = ChatOllama(
                model=llm_model,
                temperature=temperature
            )

            if self.debug:
                print(f"✅ Ollama LLM initialized")
                print()

        elif llm_provider in ["openai", "openai-compatible"]:
            if not llm_api_key:
                raise ValueError(f"API key is required for {llm_provider} provider. Use --llm-api-key or set environment variable.")

            # Determine base URL
            if llm_provider == "openai":
                final_base_url = llm_base_url or "https://api.openai.com/v1"
                provider_name = "OpenAI"
            else:
                if not llm_base_url:
                    raise ValueError("llm_base_url is required for openai-compatible provider")
                final_base_url = llm_base_url
                provider_name = "OpenAI-Compatible API"

            if self.debug:
                print(f"   Provider: {provider_name}")
                print(f"   Base URL: {final_base_url}")
                print(f"   Model: {llm_model}")
                print(f"   API Key: {'*' * (len(llm_api_key) - 4) + llm_api_key[-4:] if len(llm_api_key) > 4 else '****'}")

            # Create LLM instance
            kwargs = {
                "model": llm_model,
                "temperature": temperature,
                "api_key": llm_api_key,
                "base_url": final_base_url
            }

            self.llm = ChatOpenAI(**kwargs)

            if self.debug:
                print(f"✅ {provider_name} LLM initialized")
                print()
        else:
            raise ValueError(
                f"Unknown LLM provider: {llm_provider}. "
                f"Use 'ollama', 'openai', or 'openai-compatible'"
            )

        # Create tools using the new @tool decorator approach
        if self.debug:
            print("🔧 Creating UAV control tools...")
        self.tools = create_uav_tools(self.client)
        if self.debug:
            print(f"✅ Created {len(self.tools)} tools")
            print(f"   Tools: {', '.join([tool.name for tool in self.tools[:5]])}...")
            print()

        # Create prompt template
        if self.debug:
            print("📝 Creating agent prompt template...")
        self.prompt = self._create_prompt()
        if self.debug:
            print("✅ Prompt template created")
            print()

        # Create ReAct agent
        if self.debug:
            print("🤖 Creating ReAct agent...")
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        if self.debug:
            print("✅ ReAct agent created")
            print()

        # Create agent executor with improved error handling
        if self.debug:
            print("⚙️  Creating agent executor...")
            print(f"   Max iterations: 20")
            print(f"   Verbose mode: {verbose}")

        # Custom error handler to help LLM fix formatting issues
        def handle_parsing_error(error) -> str:
            """Provide helpful feedback when Action Input parsing fails"""
            return PARSING_ERROR_TEMPLATE.format(error=str(error))

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=verbose,
            handle_parsing_errors=handle_parsing_error,
            max_iterations=50,  # Increased for complex tasks
            return_intermediate_steps=True,
            early_stopping_method="generate"  # Better handling of completion
        )
        if self.debug:
            print("✅ Agent executor created")
            print()

        # Session context
        if self.debug:
            print("🔄 Refreshing session context...")
        self.session_context = {}
        self.refresh_session_context()

        if self.debug:
            print("="*60)
            print("✅ UAV Agent Initialization Complete!")
            print("="*60)
            print()

    def _create_prompt(self) -> PromptTemplate:
        """Create the agent prompt template"""
        return PromptTemplate(
            template=AGENT_PROMPT,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([
                    f"- {tool.name}: {tool.description}"
                    for tool in self.tools
                ]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

    def refresh_session_context(self):
        """Refresh session context information"""
        try:
            session = self.client.get_current_session()
            self.session_context = {
                'session_id': session.get('id'),
                'task_type': session.get('task'),
                'task_description': session.get('task_description'),
                'status': session.get('status')
            }
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not refresh session context: {e}")

    def get_session_summary(self) -> str:
        """Get a summary of the current session"""
        try:
            session = self.client.get_current_session()
            progress = self.client.get_task_progress()
            drones = self.client.list_drones()

            summary = f"""
=== Current Session Summary ===
Session: {session.get('name', 'Unknown')}
Task: {session.get('task', 'Unknown')} - {session.get('task_description', '')}
Status: {session.get('status', 'Unknown')}

Progress: {progress.get('progress_percentage', 0)}% ({progress.get('status_message', 'Unknown')})
Completed: {progress.get('is_completed', False)}

Drones: {len(drones)} available
"""
            for drone in drones:
                summary += f"  - {drone.get('name')} ({drone.get('id')}): {drone.get('status')}, Battery: {drone.get('battery_level', 0):.1f}%\n"

            return summary.strip()
        except Exception as e:
            return f"Error getting session summary: {e}"

    def execute(self, command: str) -> Dict[str, Any]:
        """
        Execute a natural language command

        Args:
            command: Natural language command from user

        Returns:
            Dictionary with 'output', 'intermediate_steps', and 'success' keys
        """
        if self.debug:
            print(f"\n{'='*60}")
            print(f"🎯 Executing Command")
            print(f"{'='*60}")
            print(f"Command: {command}")
            print(f"{'='*60}\n")

        try:
            if self.debug:
                print("🔄 Invoking agent executor...")

            result = self.agent_executor.invoke({"input": command})

            if self.debug:
                print(f"\n{'='*60}")
                print("✅ Command Execution Complete")
                print(f"{'='*60}")
                print(f"Success: True")
                print(f"Intermediate steps: {len(result.get('intermediate_steps', []))}")
                print(f"{'='*60}\n")

            return {
                'success': True,
                'output': result.get('output', ''),
                'intermediate_steps': result.get('intermediate_steps', [])
            }
        except Exception as e:
            if self.debug:
                print(f"\n{'='*60}")
                print("❌ Command Execution Failed")
                print(f"{'='*60}")
                print(f"Error: {str(e)}")
                print(f"{'='*60}\n")

            return {
                'success': False,
                'output': f"Error executing command: {str(e)}",
                'intermediate_steps': []
            }

    def run_interactive(self):
        """Run the agent in interactive mode"""
        print("\n" + "="*60)
        print("🚁 UAV Control Agent - Interactive Mode")
        print("="*60)
        print("\nType 'quit', 'exit', or 'q' to stop")
        print("Type 'status' to see session summary")
        print("Type 'help' for example commands\n")

        # Show initial session summary
        print(self.get_session_summary())
        print("\n" + "-"*60 + "\n")

        while True:
            try:
                user_input = input("\n🎮 Command: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'status':
                    print(self.get_session_summary())
                    continue

                if user_input.lower() == 'help':
                    self._print_help()
                    continue

                # Execute command
                print("\n🤖 Processing...\n")
                result = self.execute(user_input)

                if result['success']:
                    print(f"\n✅ {result['output']}\n")
                else:
                    print(f"\n❌ {result['output']}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def _print_help(self):
        """Print example commands"""
        help_text = """
Example Commands:
==================

Information:
- "What drones are available?"
- "Show me the current mission status"
- "What targets do I need to visit?"
- "Check the weather conditions"
- "What's the task progress?"

Basic Control:
- "Take off drone-abc123 to 15 meters"
- "Move drone-abc123 to coordinates x=100, y=50, z=20"
- "Land drone-abc123"
- "Return all drones home"

Mission Execution:
- "Visit all targets with the first drone"
- "Search the area with available drones"
- "Complete the mission task"
- "Patrol the assigned areas"

Safety:
- "Check if there are obstacles between (0,0,10) and (100,100,10)"
- "What's nearby drone-abc123?"
- "Check battery levels"

Smart Commands:
- "Take photos at all target locations"
- "Charge any drones with low battery"
- "Survey all targets and return home"
"""
        print(help_text)


def main():
    """Main entry point"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="UAV Control Agent - Natural Language Drone Control"
    )
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='UAV API base URL'
    )
    parser.add_argument(
        '--uav-api-key',
        default=None,
        help='API key for UAV server (defaults to USER role if not provided, or set UAV_API_KEY env var)'
    )
    parser.add_argument(
        '--llm-provider',
        default=None,
        choices=['ollama', 'openai', 'openai-compatible'],
        help='LLM provider (ollama, openai, or openai-compatible for DeepSeek, etc.)'
    )
    parser.add_argument(
        '--llm-model',
        default=None,
        help='LLM model name (e.g., llama2, gpt-4o-mini, deepseek-chat)'
    )
    parser.add_argument(
        '--llm-api-key',
        default=None,
        help='API key for LLM provider (or set via environment variable)'
    )
    parser.add_argument(
        '--llm-base-url',
        default=None,
        help='Custom base URL for LLM API (required for openai-compatible providers)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='LLM temperature (0.0-1.0)'
    )
    parser.add_argument(
        '--command', '-c',
        default=None,
        help='Single command to execute (non-interactive)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce verbosity'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output for connection and setup info'
    )
    parser.add_argument(
        '--no-prompt',
        action='store_true',
        help='Skip interactive provider/model selection (use command line args or defaults)'
    )

    args = parser.parse_args()

    # Determine if we should prompt for config
    should_prompt = (
        not args.no_prompt and
        not args.command and  # Only prompt in interactive mode
        args.llm_provider is None and  # No provider specified
        args.llm_model is None  # No model specified
    )

    # Get configuration from user prompt or command line
    if should_prompt:
        config = prompt_user_for_llm_config()
        if config:
            llm_provider = config.get('llm_provider', 'ollama')
            llm_model = config.get('llm_model', 'llama2')
            llm_base_url = config.get('llm_base_url')
            llm_api_key = config.get('llm_api_key')
        else:
            # Fallback to defaults
            llm_provider = 'ollama'
            llm_model = 'llama2'
            llm_base_url = None
            llm_api_key = None
    else:
        # Use command line arguments or defaults
        llm_provider = args.llm_provider or 'ollama'
        llm_model = args.llm_model or 'llama2'
        llm_base_url = args.llm_base_url
        llm_api_key = args.llm_api_key

    # Get LLM API key from args or environment if not set
    if not llm_api_key:
        llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

    # Get UAV API key from args or environment
    uav_api_key = args.uav_api_key or os.getenv("UAV_API_KEY")

    # Create agent
    try:
        agent = UAVControlAgent(
            base_url=args.base_url,
            uav_api_key=uav_api_key,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            temperature=args.temperature,
            verbose=not args.quiet,
            debug=args.debug
        )
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        print("\nMake sure:")
        print("  - Ollama is running (if using --llm-provider ollama)")
        print("  - OPENAI_API_KEY is set (if using --llm-provider openai)")
        print("  - UAV API server is accessible")
        return 1

    if args.command:
        # Single command mode
        result = agent.execute(args.command)
        print(result['output'])
        return 0 if result['success'] else 1
    else:
        # Interactive mode
        agent.run_interactive()
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
