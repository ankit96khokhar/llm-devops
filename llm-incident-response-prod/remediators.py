
from k8s_client import delete_pod, restart_deployment

def remediate_issue(pod, issue):
    name = pod.metadata.name
    ns = pod.metadata.namespace
    t = issue["type"]

    owner = pod.metadata.owner_references
    if owner:
        owner_kind = owner[0].kind
        owner_name = owner[0].name
    else:
        owner_kind = None
        owner_name = None

    if t in ["CrashLoopBackOff", "OOMKilled"]:
        if owner_kind == "ReplicaSet":
            print(f"🔁 Restarting deployment for pod {name}")
            restart_deployment(ns, owner_name)
        else:
            print(f"🛠 Restarting pod {name}")
            delete_pod(name, ns)

    elif t == "ImagePullBackOff":
        print(f"⚠️ Image issue in {name}, manual intervention needed")

    elif t == "Pending":
        print(f"⏳ Pod {name} is Pending, resource or scheduling issue")
