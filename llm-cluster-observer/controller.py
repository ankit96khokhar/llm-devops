
import time
import requests
from k8s_client import list_all_pods, get_pod_logs

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "llama3"

def build_prompt(pod, logs):
    return f"""
You are an SRE assistant. Analyze the following Kubernetes pod issue and suggest what should be done. Do NOT perform actions.

Pod: {pod.metadata.name}
Namespace: {pod.metadata.namespace}
Phase: {pod.status.phase}
Reason: {pod.status.reason}
Restarts: {sum([c.restart_count for c in (pod.status.container_statuses or [])])}

Recent Logs:
{logs}

Give a diagnosis and recommended action.
"""

def ask_llm(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    return r.json().get("response", "")

def main():
    print("🧠 LLM Cluster Observer Started (Detection + Suggestions Only)")

    while True:
        pods = list_all_pods()
        for pod in pods:
            if pod.status.phase in ["Failed", "Pending"] or any(
                c.state and c.state.waiting for c in (pod.status.container_statuses or [])
            ):
                logs = get_pod_logs(pod.metadata.name, pod.metadata.namespace)
                prompt = build_prompt(pod, logs)
                print(f"🚨 Issue detected in {pod.metadata.namespace}/{pod.metadata.name}")
                suggestion = ask_llm(prompt)
                print("🧠 LLM Suggestion:")
                print(suggestion)
                print("=" * 80)

        time.sleep(20)

if __name__ == "__main__":
    main()
