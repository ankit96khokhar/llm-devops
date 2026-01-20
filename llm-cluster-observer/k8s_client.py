
from kubernetes import client, config

try:
    config.load_incluster_config()
except:
    config.load_kube_config()

v1 = client.CoreV1Api()

def list_all_pods():
    return v1.list_pod_for_all_namespaces().items

def get_pod_logs(name, namespace):
    try:
        return v1.read_namespaced_pod_log(name=name, namespace=namespace, tail_lines=20)
    except:
        return "No logs available"
