import os

from hr_agent_starter import build_graph, run_case

# Deliberately request a nonexistent model.
os.environ["GROQ_MODEL"] = "nonexistent-lab-test-model"

graph = build_graph()

result, status = run_case(
    graph,
    "LLM_FAILURE",
    "Хочу взять отпуск",
)

expected_route = ["intake", "classify", "escalate", "respond"]

ok = (
    status == "DONE"
    and result["route_log"] == expected_route
    and bool(result.get("error"))
)

print("Fallback test:", "PASS" if ok else "FAIL")
print("Route:", " -> ".join(result["route_log"]))
print("Error:", result.get("error"))
print("Response:", result.get("response"))