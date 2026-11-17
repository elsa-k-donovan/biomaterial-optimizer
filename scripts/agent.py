"""Ask the agent a question about seaweed films.

Needs ANTHROPIC_API_KEY in .env, and build_db.py + build_dataset.py run first.

Usage:
    uv run --env-file .env scripts/agent.py "How can I make an alginate film more water-resistant?"
"""

import os
import sys

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai.messages import ToolCallPart

from biomaterial_optimizer.agent import Deps, agent


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    result = agent.run_sync(sys.argv[1], deps=Deps())
    calls = [
        part.tool_name
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolCallPart)
    ]
    print(result.output)
    print(f"\n[tools used: {', '.join(calls) or 'none'}; {result.usage()}]")


if __name__ == "__main__":
    main()
