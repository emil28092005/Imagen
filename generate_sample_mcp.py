#!/usr/bin/env python3
"""Generate a sample dataset through the MCP server."""

import asyncio
import os
import sys

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_SPECS = [
    {
        "prompt": "a brave knight in shining armor holding a sword",
        "output_path": "tetra/knight_01.png",
        "seed": 42,
    },
    {
        "prompt": "an elven archer with a longbow and green cloak",
        "output_path": "tetra/archer_01.png",
        "seed": 43,
    },
    {
        "prompt": "a robed mage casting a blue fireball",
        "output_path": "tetra/mage_01.png",
        "seed": 44,
    },
    {
        "prompt": "a peasant worker carrying a wooden hammer",
        "output_path": "tetra/worker_01.png",
        "seed": 45,
    },
    {
        "prompt": "a crystal golem with geometric facets",
        "output_path": "tetra/golem_01.png",
        "seed": 46,
    },
    {
        "prompt": "a chaos demon with horns and jagged armor",
        "output_path": "tetra/demon_01.png",
        "seed": 47,
    },
    {
        "prompt": "an ancient treant with glowing green eyes",
        "output_path": "tetra/treant_01.png",
        "seed": 48,
    },
    {
        "prompt": "a skeleton warrior with a rusted sword",
        "output_path": "tetra/skeleton_01.png",
        "seed": 49,
    },
]


async def main():
    server_path = os.path.join(BASE_DIR, "server.py")
    params = StdioServerParameters(
        command=sys.executable,
        args=[server_path],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Calling batch_generate...")
            result = await session.call_tool("batch_generate", {"specs": SAMPLE_SPECS})

            print("\nResults:")
            for item in result.content:
                if item.type == "text":
                    print(item.text)

            stats = await session.call_tool("db_stats", {})
            print("\nDB stats:")
            for item in stats.content:
                if item.type == "text":
                    print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
