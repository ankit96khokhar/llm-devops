#!/bin/bash
set -e

echo "🤖 Setting up LLM-Powered Kubernetes Incident Response System"

# Check for LLM configuration
LLM_TYPE=${LLM_TYPE:-"rules"}
echo "🧠 Using LLM type: $LLM_TYPE"

if [ "$LLM_TYPE" = "ollama" ]; then
    echo "🦙 Ollama mode - make sure 'ollama serve' is running"
    echo "📝 Models available: $(ollama list 2>/dev/null | grep -v NAME || echo 'Run: ollama pull llama3.2')"
elif [ "$LLM_TYPE" = "huggingface" ]; then
    echo "🤗 Hugging Face mode - local transformers"
else
    echo "🆓 Rule-based mode - no external dependencies"
fi

# Check if we're in the right directory
if [ ! -f "incident_analyzer.py" ]; then
    echo "❌ Please run this script from the llm directory"
    exit 1
fi

# Check if minikube is running
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Kubernetes cluster not accessible. Make sure minikube is running:"
    echo "   minikube start"
    exit 1
fi

echo "✅ Kubernetes cluster accessible"

# Create namespace and RBAC
echo "📋 Creating namespace and RBAC configuration..."
kubectl apply -f kubernetes-deployment.yaml

# Wait for namespace to be ready
kubectl wait --for=condition=Ready namespace/llm-incident-response --timeout=30s

echo "✅ Namespace and RBAC created"

# Check if we should build Docker image or use ConfigMap approach
if [ -f "Dockerfile" ] && command -v docker &> /dev/null; then
    echo "🐳 Docker available - building proper image..."
    
    # Use minikube's Docker daemon if available
    if command -v minikube &> /dev/null && minikube status | grep -q "Running"; then
        echo "🚀 Using minikube Docker daemon..."
        eval $(minikube docker-env)
    fi
    
    # Build Docker image
    docker build -t llm-incident-response:latest . || {
        echo "❌ Docker build failed, falling back to ConfigMap approach"
        USE_CONFIGMAP=true
    }
    
    if [ "$USE_CONFIGMAP" != "true" ]; then
        echo "✅ Docker image built successfully"
        USE_IMAGE=true
    fi
else
    echo "🔧 No Docker or Dockerfile - using ConfigMap approach"
    USE_CONFIGMAP=true
fi

if [ "$USE_CONFIGMAP" = "true" ]; then
    # Create ConfigMap with the actual code (fallback approach)
    echo "📦 Creating application ConfigMap..."
    kubectl create configmap llm-incident-response-code \
        --from-file=incident_analyzer.py \
        --from-file=argocd_integration.py \
        -n llm-incident-response \
        --dry-run=client -o yaml | kubectl apply -f -
    
    echo "✅ Application code deployed via ConfigMap"
fi

echo "🆓 No external API costs - all analysis runs locally!"

# Create environment-aware secret
kubectl create secret generic llm-incident-response-secrets \
    --from-literal=LLM_TYPE="$LLM_TYPE" \
    --from-literal=OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}" \
    --from-literal=ARGOCD_USERNAME="admin" \
    --from-literal=ARGOCD_PASSWORD="${ARGOCD_PASSWORD:-admin}" \
    -n llm-incident-response \
    --dry-run=client -o yaml | kubectl apply -f -

# Create ConfigMap with the actual code
echo "📦 Creating application ConfigMap..."
kubectl create configmap llm-incident-response-code \
    --from-file=incident_analyzer.py \
    --from-file=argocd_integration.py \
    -n llm-incident-response \
    --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Application code deployed"

# Deploy the application
echo "🚀 Deploying LLM Incident Response System..."

# Create deployment based on whether we have custom image or not
if [ "$USE_IMAGE" = "true" ]; then
    echo "🐳 Using custom Docker image deployment..."
    
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
        image: llm-incident-response:latest
        imagePullPolicy: Never  # Use local image in minikube
        
        env:
        - name: LLM_TYPE
          valueFrom:
            secretKeyRef:
              name: llm-incident-response-secrets
              key: LLM_TYPE
              optional: true
        - name: OLLAMA_HOST
          valueFrom:
            secretKeyRef:
              name: llm-incident-response-secrets
              key: OLLAMA_HOST
              optional: true
        - name: ARGOCD_USERNAME
          valueFrom:
            secretKeyRef:
              name: llm-incident-response-secrets
              key: ARGOCD_USERNAME
              optional: true
        - name: ARGOCD_PASSWORD
          valueFrom:
            secretKeyRef:
              name: llm-incident-response-secrets
              key: ARGOCD_PASSWORD
              optional: true
        - name: ARGOCD_SERVER
          value: "https://argocd-server.argocd.svc.cluster.local"
        
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
            
        livenessProbe:
          exec:
            command: ["python", "-c", "print('healthy')"]
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          
        readinessProbe:
          exec:
            command: ["python", "-c", "print('ready')"]
          initialDelaySeconds: 30
          periodSeconds: 10
EOF

else
    echo "📦 Using ConfigMap-based deployment..."
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
        command: ["/bin/bash"]
        args: 
        - -c
        - |
          echo "Installing Python dependencies..."
          pip install --no-cache-dir \
            kubernetes==28.1.0 \
            requests==2.31.0 \
            pyyaml==6.0.1 \
            aiohttp==3.9.1 \
            structlog==23.2.0
          
          echo "🆓 Starting FREE Kubernetes Incident Response System..."
          echo "📊 Using intelligent rule-based analysis (no APIs required)"
          cd /app && python incident_analyzer.py
        
        env:
        - name: LLM_TYPE
          valueFrom:
            secretKeyRef:
              name: llm-incident-response-secrets
              key: LLM_TYPE
              optional: true
        - name: OLLAMA_HOST
          valueFrom:
            secretKeyRef:
              name: llm-incident-response-secrets
              key: OLLAMA_HOST
              optional: true
        - name: ARGOCD_USERNAME
          valueFrom:
            secretKeyRef:
              name: llm-incident-response-secrets
              key: ARGOCD_USERNAME
              optional: true
        - name: ARGOCD_PASSWORD
          valueFrom:
            secretKeyRef:
              name: llm-incident-response-secrets
              key: ARGOCD_PASSWORD
              optional: true
        - name: ARGOCD_SERVER
          value: "https://argocd-server.argocd.svc.cluster.local"
        
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /app/config
        
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
            
        livenessProbe:
          exec:
            command: ["python", "-c", "print('healthy')"]
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          
        readinessProbe:
          exec:
            command: ["python", "-c", "print('ready')"]
          initialDelaySeconds: 30
          periodSeconds: 10
      
      volumes:
      - name: app-code
        configMap:
          name: llm-incident-response-code
          defaultMode: 0755
      - name: config
        configMap:
          name: llm-incident-response-config
EOF

echo "✅ Deployment created"

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=Available deployment/llm-incident-response \
    -n llm-incident-response --timeout=300s

echo "✅ LOCAL LLM Incident Response System deployed successfully!"
echo "🆓 No external API costs - runs completely locally!"

# Show status
echo ""
echo "📊 Current Status:"
kubectl get all -n llm-incident-response

echo ""
echo "📝 To view logs:"
echo "   kubectl logs -f deployment/llm-incident-response -n llm-incident-response"

echo ""
echo "🔍 To test the system:"
echo "   ./create_test_scenarios.sh"

echo ""
echo "🎯 System is monitoring with LLM type: $LLM_TYPE"

# Show configuration instructions
echo ""
echo "🤖 LLM Options Available:"
echo "   🆓 rules: Fast rule-based analysis (default)"
echo "   🦙 ollama: Local Llama/Mistral models via Ollama"  
echo "   🤗 huggingface: Local transformers models"

if [ "$LLM_TYPE" = "ollama" ]; then
    echo ""
    echo "🦙 Ollama Setup:"
    echo "   - Make sure ollama is running: ollama serve"
    echo "   - Available models: ollama list"
    echo "   - Recommended: ollama pull llama3.2"
elif [ "$LLM_TYPE" = "huggingface" ]; then
    echo ""
    echo "🤗 Hugging Face Setup:"
    echo "   - Models download automatically on first use"
    echo "   - Requires ~2GB disk space"
    echo "   - GPU recommended but not required"
fi

echo ""
echo "🔧 To change LLM type:"
echo "   export LLM_TYPE='ollama|huggingface|rules'"
echo "   kubectl delete pod -l app=llm-incident-response -n llm-incident-response"

echo ""
echo "✨ Setup complete! Advanced LOCAL LLM analysis ready."
