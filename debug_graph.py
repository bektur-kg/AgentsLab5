from langgraph.types import Command

from hr_agent_starter import build_graph

graph = build_graph()
cfg = {"configurable": {"thread_id": "DEBUG_T06"}}


def show_updates(inputs):
    for update in graph.stream(inputs, cfg, stream_mode="updates"):
        print(update)


print("=== Initial request ===")
show_updates({"request": "Помогите", "route_log": []})

print("\nPaused at:", graph.get_state(cfg).next)

print("\n=== First clarification ===")
show_updates(Command(resume="не знаю"))

print("\nPaused at:", graph.get_state(cfg).next)

print("\n=== Second clarification ===")
show_updates(Command(resume="просто так"))

final = graph.get_state(cfg)
print("\nFinal next:", final.next)
print("Route:", " -> ".join(final.values["route_log"]))

# History is returned newest first; reverse it for chronological output.
history = list(graph.get_state_history(cfg))

print("\n=== Checkpoint history ===")
for snapshot in reversed(history):
    print(
        "step:", snapshot.metadata.get("step"),
        "| next:", snapshot.next,
        "| writes:", snapshot.metadata.get("writes"),
    )

print("\nTotal snapshots:", len(history))