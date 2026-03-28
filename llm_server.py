"""
LLM proxy server — wraps the `claude -p` CLI as an HTTP endpoint.

This avoids needing an API key by piggy-backing on the locally
authenticated Claude Code CLI. Any service that needs LLM generation
can POST a prompt to this server and get back Claude's response.

Usage:
    uvicorn llm_server:app --port 8002
"""

import subprocess
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Claude CLI LLM Proxy")


class PromptRequest(BaseModel):
    """Request body for the /generate endpoint."""
    # The prompt text to send to Claude
    prompt: str
    # Optional: max output tokens (not directly supported by CLI, reserved for future use)
    max_tokens: int | None = None


class GenerateResponse(BaseModel):
    """Response body from the /generate endpoint."""
    response: str


def run_claude(prompt: str) -> str:
    """
    Execute the `claude -p` CLI with the given prompt and return the response.

    The `-p` flag tells Claude Code to run in non-interactive "pipe" mode:
    it reads the prompt, generates a response, and exits.

    Args:
        prompt: The full prompt string to send to Claude.

    Returns:
        Claude's text response, with markdown code fences stripped.

    Raises:
        HTTPException: If the claude CLI fails or is not found.
    """
    try:
        # Run `claude -p <prompt>` as a subprocess.
        # capture_output=True captures both stdout and stderr.
        # text=True returns strings instead of bytes.
        # timeout=120 prevents hanging if Claude takes too long.
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # If the CLI returned a non-zero exit code, something went wrong
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"claude CLI error: {result.stderr.strip()}",
            )

        # Clean up the response — strip whitespace and remove markdown
        # code fences that Claude sometimes wraps around its output
        response = result.stdout.strip()
        response = re.sub(r"^```(?:json)?\s*\n?", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\n?```\s*$", "", response, flags=re.IGNORECASE)
        return response.strip()

    except FileNotFoundError:
        # The `claude` command is not installed or not on PATH
        raise HTTPException(
            status_code=500,
            detail="claude CLI not found. Make sure Claude Code is installed and on your PATH.",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="claude CLI timed out after 120 seconds.",
        )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: PromptRequest):
    """
    Send a prompt to Claude via the CLI and return the generated response.

    POST /generate
    Body: {"prompt": "your prompt here"}
    Returns: {"response": "Claude's answer"}
    """
    response = run_claude(req.prompt)
    return GenerateResponse(response=response)


@app.get("/health")
def health():
    return {"status": "ok"}
