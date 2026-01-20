
import time
from detectors import detect_pod_issues
from remediators import remediate_issue
from k8s_client import list_all_pods
from state import should_act, record_action

SCAN_INTERVAL = 10

def main():
    print("🚑 Auto-Healing Controller Started (Production Rules Mode)")
    while True:
        pods = list_all_pods()
        for pod in pods:
            issue = detect_pod_issues(pod)
            if issue:
                key = f"{pod.metadata.namespace}/{pod.metadata.name}:{issue['type']}"
                if should_act(key):
                    print(f"🚨 Detected: {issue['type']} in {pod.metadata.name}")
                    remediate_issue(pod, issue)
                    record_action(key)
                else:
                    print(f"⏳ Cooldown active for {pod.metadata.name}")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
