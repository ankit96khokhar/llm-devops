#!/bin/bash

echo "🧪 Creating test scenarios for LLM Incident Response System"

# Test 1: ImagePullBackOff scenario
echo "Test 1: Creating pod with non-existent image (ImagePullBackOff)"
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-imagepullbackoff
  namespace: default
  labels:
    test: incident-response
spec:
  containers:
  - name: failing-container
    image: nonexistent-registry/bad-image:latest
    ports:
    - containerPort: 8080
EOF

# Test 2: CrashLoopBackOff scenario  
echo "Test 2: Creating crashing pod (CrashLoopBackOff)"
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-crashloop
  namespace: default
  labels:
    test: incident-response
spec:
  restartPolicy: Always
  containers:
  - name: crashing-container
    image: busybox
    command: ["sh", "-c", "echo 'Starting up...'; sleep 5; echo 'Crashing now!'; exit 1"]
EOF

# Test 3: Memory limit exceeded (OOMKilled)
echo "Test 3: Creating memory-exhausted pod (OOMKilled)"
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-oomkilled
  namespace: default
  labels:
    test: incident-response
spec:
  containers:
  - name: memory-hog
    image: busybox
    command: ["sh", "-c", "echo 'Allocating memory...'; dd if=/dev/zero of=/tmp/memory.dat bs=1M count=200; sleep 3600"]
    resources:
      limits:
        memory: "50Mi"
      requests:
        memory: "10Mi"
EOF

# Test 4: Problematic deployment
echo "Test 4: Creating deployment with issues"
cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-problematic-deployment
  namespace: default
  labels:
    test: incident-response
spec:
  replicas: 3
  selector:
    matchLabels:
      app: problematic-app
  template:
    metadata:
      labels:
        app: problematic-app
    spec:
      containers:
      - name: app
        image: nginx:1.20
        ports:
        - containerPort: 80
        # Intentionally bad configuration
        env:
        - name: INVALID_CONFIG
          value: "this-will-cause-issues"
        command: ["sh", "-c", "echo 'App starting...'; sleep 10; nginx -t || exit 1; nginx -g 'daemon off;'"]
        resources:
          limits:
            memory: "64Mi"
            cpu: "50m"
          requests:
            memory: "32Mi"
            cpu: "10m"
EOF

# Test 5: Pending pod (resource constraints)
echo "Test 5: Creating pod with impossible resource requests"
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-pending-resources
  namespace: default
  labels:
    test: incident-response
spec:
  containers:
  - name: resource-hungry
    image: nginx
    resources:
      requests:
        memory: "100Gi"  # Impossible request for typical dev clusters
        cpu: "50"
EOF

# Test 6: Working deployment that we'll break
echo "Test 6: Creating working deployment for rollback testing"
cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-rollback-deployment
  namespace: default
  labels:
    test: incident-response
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rollback-test
  template:
    metadata:
      labels:
        app: rollback-test
    spec:
      containers:
      - name: web
        image: nginx:1.20
        ports:
        - containerPort: 80
EOF

echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=60s deployment/test-rollback-deployment

# Now break it with a bad image
echo "Breaking deployment with bad image..."
kubectl set image deployment/test-rollback-deployment web=nginx:nonexistent-tag

echo ""
echo "✅ Test scenarios created! The system should detect and remediate these issues."
echo ""
echo "Monitor the incident response system:"
echo "kubectl logs -f deployment/llm-incident-response -n llm-incident-response"
echo ""
echo "Check pod status:"
echo "kubectl get pods -l test=incident-response"
echo ""
echo "View events:"
echo "kubectl get events --sort-by='.lastTimestamp'"
echo ""
echo "🕐 Wait 2-3 minutes for the system to detect and process incidents..."

# Optional: Create a monitoring script
cat << 'EOF' > monitor_tests.sh
#!/bin/bash
echo "📊 Monitoring Test Scenarios"
echo "=========================="

while true; do
    clear
    echo "🕐 $(date)"
    echo ""
    
    echo "📋 Test Pods Status:"
    kubectl get pods -l test=incident-response -o wide
    echo ""
    
    echo "🚀 Test Deployments:"
    kubectl get deployments -o wide | grep test-
    echo ""
    
    echo "📝 Recent Events (last 5):"
    kubectl get events --sort-by='.lastTimestamp' --field-selector involvedObject.name!=test-pending-resources | tail -5
    echo ""
    
    echo "🔍 Incident Response System Status:"
    kubectl get pods -n llm-incident-response
    echo ""
    
    echo "Press Ctrl+C to stop monitoring..."
    sleep 10
done
EOF

chmod +x monitor_tests.sh

echo "📱 Created monitoring script: ./monitor_tests.sh"
echo "Run it to continuously monitor the test scenarios."
