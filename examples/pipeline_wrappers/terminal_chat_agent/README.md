# Chat with files in a server workspace

Deploy one Haystack Agent with `list_files`, `read_file`, and `write_file`, then use terminal chat to work with text files.
The wrapper owns the Agent, its model, and its tools. `file_tools.py` also works with an ordinary Python Agent without terminal chat.

Run these commands from the Hayhooks repository root.
Install terminal chat in the Python environment where you run the client:

```sh
pip install 'hayhooks[chat]'
```

The server needs Hayhooks and the model credentials. The server and client can use separate Python environments.

Create a dedicated demo directory owned by the server operator. Keep deployment source files in a separate directory.
Set these variables in the environment that starts the Hayhooks server.

```sh
mkdir -p /tmp/hayhooks-chat-demo
printf 'The launch review is on Friday.\n' > /tmp/hayhooks-chat-demo/notes.txt
export EXAMPLE_CHAT_WORKSPACE=/tmp/hayhooks-chat-demo
export OPENAI_API_KEY=your-api-key
export EXAMPLE_AGENT_MODEL=gpt-4.1-mini
hayhooks run
```

`EXAMPLE_CHAT_WORKSPACE` is required and must name an existing server-side directory.
The wrapper resolves it during setup, including configured directory aliases.
`EXAMPLE_AGENT_MODEL` defaults to `gpt-4.1-mini`. The Agent uses at most ten steps per request.

In another terminal, deploy both Python files together and open chat.

```sh
hayhooks pipeline deploy-files -n terminal-chat-agent examples/pipeline_wrappers/terminal_chat_agent
hayhooks chat terminal-chat-agent
```

Try these requests in order.

1. "List the workspace files."
2. "Read notes.txt."
3. "Write a short summary of notes.txt to summary.txt."
4. "Read summary.txt back."

Chat shows each tool's arguments, result, and failure status. Check the written file in the server workspace.
To try error recovery, ask to read a missing file, then ask to read `notes.txt`.
A failed tool returns an error to the Agent so it can explain the problem and continue.

Use relative paths beneath the configured directory. Tools reject absolute paths, parent traversal, symlinks in requested paths, and nonregular file targets.
`list_files` returns at most 200 immediate entries and states when it truncates the listing. It marks symlinks without following them.
Reads and write content must be UTF-8 text of at most 65,536 encoded bytes. Writes replace the full file and require an existing parent directory.
A write stages content in a sibling temporary file before replacing the destination. Validation failures leave existing files unchanged.
If a filesystem failure reports uncertain completion, inspect the file before retrying.

These containment checks are not an OS sandbox and cannot eliminate filesystem races.
Use a dedicated operator-owned demo directory without sensitive files or other processes changing its structure.
Simultaneous writes to the same path can overwrite each other.
Stopping chat reception does not roll back file changes or guarantee that server tools stop.
