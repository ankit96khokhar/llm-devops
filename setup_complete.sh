#!/bin/bash
set -e

echo "🚀 Complete Setup Guide - LLM Incident Response System"
echo "Choose your deployment method:"
echo ""
echo "1. 🐳 Docker Image (Production-like)"
echo "2. 📦 ConfigMap (Quick & Easy)"
echo ""

read -p "Enter choice (1-2): " choice

# Set LLM configuration
LLM_TYPE=${LLM_TYPE:-"rules"}
echo "🧠 Using LLM type: $LLM_TYPE"

# Check minikube
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Kubernetes cluster not accessible. Make sure minikube is running:"
    echo "   minikube start"
    exit 1
fi

echo "✅ Kubernetes cluster accessible"

# Create namespace and RBAC
echo "📋 Creating namespace and RBAC..."
kubectl apply -f kubernetes-deployment.yaml
kubectl wait --for=condition=Ready namespace/llm-incident-response --timeout=30s

# Create secrets
kubectl create secret generic llm-incident-response-secrets \
    --from-literal=LLM_TYPE="$LLM_TYPE" \
    --from-literal=OLLAMA_HOST="${OLLAMA_HOST:-http://host.minikube.internal:11434}" \
    --from-literal=ARGOCD_USERNAME="admin" \
    --from-literal=ARGOCD_PASSWORD="${ARGOCD_PASSWORD:-admin}" \
    -n llm-incident-response \
    --dry-run=client -o yaml | kubectl apply -f -

if [ "$choice" = "1" ]; then
    echo ""
    echo "🐳 DOCKER IMAGE APPROACH"
    echo "========================="
    
    # Build Docker image
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker first."
        exit 1
    fi
    
    echo "🔧 Using minikube Docker daemon..."
    eval $(minikube docker-env)
    
    echo "🏗️  Building Docker image..."
    docker build -t llm-incident-response:v1.0 .
    
    echo "✅ Image built! Available in minikube:"
    docker images | grep llm-incident-response
    
    # Deploy using custom image
    echo "🚀 Deploying with custom Docker image..."
    cat << 'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-incident-response
  namespace: llm-incident-response
  labels:
    app: llm-incident-response
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-incident-response
  template:
    metadata:
      labels:
        app: llm-incident-response
    spec:
      serviceAccountName: llm-incident-response
      containers:
      - name: incident-analyzer
        image: llm-incident-response:v1.0
        imagePullPolicy: Never
        env:
        - name: LLM_TYPE
          value: "PLACEHOLDER_LLM_TYPE"
        - name: OLLAMA_HOST  
          value: "PLACEHOLDER_OLLAMA_HOST"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
EOF
    
    # Replace placeholders with actual values
    kubectl patch deployment llm-incident-response -n llm-incident-response --type='json' -p='[
        {"op": "replace", "path": "/spec/template/spec/containers/0/env/0/value", "value": "'$LLM_TYPE'"},
        {"op": "replace", "path": "/spec/template/spec/containers/0/env/1/value", "value": "'${OLLAMA_HOST:-http://host.minikube.internal:11434}'"}
    ]'
    
else
    echo ""
    echo "📦 CONFIGMAP APPROACH"
    echo "====================="
    
    # Create ConfigMap with source code
    echo "📁 Creating ConfigMap with application code..."
    kubectl create configmap llm-incident-response-code \
        --from-file=incident_analyzer.py \
        --from-file=argocd_integration.py \
        -n llm-incident-response \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy using ConfigMap
    echo "🚀 Deploying with ConfigMap approach..."
    cat << 'EOF' | kubectl apply -f -
apiVersion: apps/v1  
kind: Deployment
metadata:
  name: llm-incident-response
  namespace: llm-incident-response
  labels:
    app: llm-incident-response
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-incident-response
  template:
    metadata:
      labels:
        app: llm-incident-response
    spec:
      serviceAccountName: llm-incident-response
      containers:
      - name: incident-analyzer
        image: python:3.11-slim
        command: ["/bin/bash", "-c"]
        args: 
        - |
          pip install --no-cache-dir kubernetes==28.1.0 requests==2.31.0 pyyaml==6.0.1 aiohttp==3.9.1 structlog==23.2.0
          cd /app && python incident_analyzer.py
        env:
        - name: LLM_TYPE
          value: "PLACEHOLDER_LLM_TYPE"
        - name: OLLAMA_HOST
          value: "PLACEHOLDER_OLLAMA_HOST"
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi" 
            cpu: "500m"
      volumes:
      - name: app-code
        configMap:
          name: llm-incident-response-code
          defaultMode: 0755
EOF
    
    # Replace placeholders
    kubectl patch deployment llm-incident-response -n llm-incident-response --type='json' -p='[
        {"op": "replace", "path": "/spec/template/spec/containers/0/env/0/value", "value": "'$LLM_TYPE'"},
        {"op": "replace", "path": "/spec/template/spec/containers/0/env/1/value", "value": "'${OLLAMA_HOST:-http://host.minikube.internal:11434}'"}
    ]'
fi

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=Available deployment/llm-incident-response \
    -n llm-incident-response --timeout=300s

echo ""
echo "✅ Deployment successful!"

# Show status
echo ""
echo "📊 Current Status:"
kubectl get all -n llm-incident-response

echo ""
echo "📝 To view logs:"
echo "   kubectl logs -f deployment/llm-incident-response -n llm-incident-response"

echo ""
echo "🧪 To test the system:"
echo "   ./create_test_scenarios.sh"

echo ""
if [ "$choice" = "1" ]; then
    echo "🐳 DOCKER APPROACH COMPLETE!"
    echo "   ✅ Code is baked into the Docker image"
    echo "   ✅ Production-ready deployment"
    echo "   ✅ Faster startup (no pip install)"
    echo ""
    echo "🔄 To update code:"
    echo "   1. Modify Python files"
    echo "   2. Run: eval \$(minikube docker-env) && docker build -t llm-incident-response:v1.1 ."
    echo "   3. Update image: kubectl set image deployment/llm-incident-response incident-analyzer=llm-incident-response:v1.1 -n llm-incident-response"
else
    echo "📦 CONFIGMAP APPROACH COMPLETE!"
    echo "   ✅ Easy development workflow"
    echo "   ✅ Quick code changes"
    echo "   ⚠️  Slower startup (installs deps each time)"
    echo ""
    echo "🔄 To update code:"
    echo "   1. Modify Python files"
    echo "   2. Run: kubectl create configmap llm-incident-response-code --from-file=incident_analyzer.py --from-file=argocd_integration.py -n llm-incident-response --dry-run=client -o yaml | kubectl apply -f -"
    echo "   3. Restart: kubectl rollout restart deployment/llm-incident-response -n llm-incident-response"
fi

echo ""
echo "🎯 System is now monitoring your cluster with LLM type: $LLM_TYPE"
