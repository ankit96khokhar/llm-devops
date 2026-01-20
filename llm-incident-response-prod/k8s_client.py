
from kubernetes import client, config

try:
    config.load_incluster_config()
except:
    config.load_kube_config()

v1 = client.CoreV1Api()
apps = client.AppsV1Api()

def list_all_pods():
    return v1.list_pod_for_all_namespaces().items

def delete_pod(name, namespace):
    v1.delete_namespaced_pod(name, namespace)

def restart_deployment(namespace, replicaset_name):
    rs = apps.list_namespaced_replica_set(namespace)
    for r in rs.items:
        if r.metadata.name == replicaset_name:
            dep_name = r.metadata.owner_references[0].name
            patch = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "restarted-at": str(int(time.time()))
                            }
                        }
                    }
                }
            }
            apps.patch_namespaced_deployment(dep_name, namespace, patch)
