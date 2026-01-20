
def detect_pod_issues(pod):
    status = pod.status
    if not status.container_statuses:
        return None

    for c in status.container_statuses:
        if c.state and c.state.waiting:
            reason = c.state.waiting.reason
            if reason in ["CrashLoopBackOff", "ImagePullBackOff"]:
                return {"type": reason}

        if c.last_state and c.last_state.terminated:
            if c.last_state.terminated.reason == "OOMKilled":
                return {"type": "OOMKilled"}

    if status.phase == "Pending":
        return {"type": "Pending"}

    return None
