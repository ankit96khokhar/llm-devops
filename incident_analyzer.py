#!/usr/bin/env python3
"""
LLM-Powered Kubernetes Incident Analyzer and Auto-Remediation Engine

This module provides real-time incident analysis and automated remediation
for Kubernetes clusters using LLM-based decision making.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml
import requests
from dataclasses import dataclass
from enum import Enum

# Kubernetes client
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

# No OpenAI needed for FREE version!


class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RemediationAction(Enum):
    RESTART_POD = "restart_pod"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    UPDATE_CONFIG = "update_config"
    DRAIN_NODE = "drain_node"
    NO_ACTION = "no_action"


@dataclass
class IncidentContext:
    """Container for incident-related data"""
    timestamp: datetime
    namespace: str
    resource_type: str
    resource_name: str
    severity: IncidentSeverity
    description: str
    logs: List[str]
    metrics: Dict[str, Any]
    events: List[Dict[str, Any]]
    resource_manifest: Optional[Dict[str, Any]] = None


@dataclass
class RemediationPlan:
    """Container for remediation actions"""
    action: RemediationAction
    target_resource: str
    target_namespace: str
    parameters: Dict[str, Any]
    reasoning: str
    confidence_score: float
    estimated_impact: str


class KubernetesMonitor:
    """Monitors Kubernetes cluster for incidents and anomalies"""
    
    def __init__(self):
        # Load Kubernetes configuration
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
            
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.metrics_client = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_pod_logs(self, namespace: str, pod_name: str, lines: int = 100) -> List[str]:
        """Retrieve recent pod logs"""
        try:
            logs = self.v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=lines,
                timestamps=True
            )
            return logs.split('\n') if logs else []
        except ApiException as e:
            self.logger.error(f"Error getting logs for pod {pod_name}: {e}")
            return []
    
    def get_pod_events(self, namespace: str, pod_name: str) -> List[Dict[str, Any]]:
        """Get recent events for a pod"""
        try:
            events = self.v1.list_namespaced_event(namespace=namespace)
            pod_events = []
            
            for event in events.items:
                if (event.involved_object.name == pod_name and 
                    event.involved_object.kind == "Pod"):
                    pod_events.append({
                        'reason': event.reason,
                        'message': event.message,
                        'type': event.type,
                        'timestamp': event.first_timestamp,
                        'count': event.count
                    })
            
            # Sort by timestamp, most recent first
            pod_events.sort(key=lambda x: x['timestamp'], reverse=True)
            return pod_events[:20]  # Return last 20 events
            
        except ApiException as e:
            self.logger.error(f"Error getting events for pod {pod_name}: {e}")
            return []
    
    def get_resource_metrics(self, namespace: str, resource_name: str, resource_type: str) -> Dict[str, Any]:
        """Get resource usage metrics"""
        metrics = {
            'cpu_usage': 'unknown',
            'memory_usage': 'unknown',
            'restart_count': 0,
            'ready_replicas': 0,
            'desired_replicas': 0
        }
        
        try:
            if resource_type == "Pod":
                pod = self.v1.read_namespaced_pod(name=resource_name, namespace=namespace)
                if pod.status.container_statuses:
                    metrics['restart_count'] = sum(
                        container.restart_count or 0 
                        for container in pod.status.container_statuses
                    )
            
            elif resource_type == "Deployment":
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=resource_name, namespace=namespace
                )
                metrics['desired_replicas'] = deployment.spec.replicas or 0
                metrics['ready_replicas'] = deployment.status.ready_replicas or 0
                
        except ApiException as e:
            self.logger.error(f"Error getting metrics for {resource_type} {resource_name}: {e}")
            
        return metrics
    
    def detect_pod_issues(self) -> List[IncidentContext]:
        """Detect problematic pods across all namespaces"""
        incidents = []
        
        try:
            # Get all pods
            pods = self.v1.list_pod_for_all_namespaces()
            
            for pod in pods.items:
                # Skip system pods unless critical
                if pod.metadata.namespace in ['kube-system', 'kube-public'] and not self._is_critical_pod(pod):
                    continue
                
                severity = self._analyze_pod_health(pod)
                if severity != IncidentSeverity.LOW:
                    # Collect incident context
                    logs = self.get_pod_logs(pod.metadata.namespace, pod.metadata.name)
                    events = self.get_pod_events(pod.metadata.namespace, pod.metadata.name)
                    metrics = self.get_resource_metrics(
                        pod.metadata.namespace, pod.metadata.name, "Pod"
                    )
                    
                    incident = IncidentContext(
                        timestamp=datetime.now(),
                        namespace=pod.metadata.namespace,
                        resource_type="Pod",
                        resource_name=pod.metadata.name,
                        severity=severity,
                        description=self._describe_pod_issue(pod),
                        logs=logs,
                        metrics=metrics,
                        events=events,
                        resource_manifest=pod.to_dict()
                    )
                    incidents.append(incident)
                    
        except ApiException as e:
            self.logger.error(f"Error detecting pod issues: {e}")
            
        return incidents
    
    def _is_critical_pod(self, pod) -> bool:
        """Determine if a system pod is critical"""
        critical_prefixes = ['coredns', 'kube-proxy', 'etcd', 'kube-apiserver']
        return any(pod.metadata.name.startswith(prefix) for prefix in critical_prefixes)
    
    def _analyze_pod_health(self, pod) -> IncidentSeverity:
        """Analyze pod health and return severity level"""
        if pod.status.phase == "Failed":
            return IncidentSeverity.HIGH
        
        if pod.status.phase == "Pending":
            # Check how long it's been pending
            creation_time = pod.metadata.creation_timestamp
            if datetime.now() - creation_time.replace(tzinfo=None) > timedelta(minutes=5):
                return IncidentSeverity.MEDIUM
        
        if pod.status.container_statuses:
            for container in pod.status.container_statuses:
                # High restart count
                if container.restart_count and container.restart_count > 5:
                    return IncidentSeverity.MEDIUM
                
                # Container not ready
                if not container.ready:
                    return IncidentSeverity.MEDIUM
                    
                # Recent restarts
                if (container.last_state and 
                    container.last_state.terminated and 
                    container.last_state.terminated.finished_at):
                    
                    last_restart = container.last_state.terminated.finished_at
                    if datetime.now() - last_restart.replace(tzinfo=None) < timedelta(minutes=10):
                        return IncidentSeverity.MEDIUM
        
        return IncidentSeverity.LOW
    
    def _describe_pod_issue(self, pod) -> str:
        """Generate human-readable description of pod issues"""
        issues = []
        
        if pod.status.phase == "Failed":
            issues.append(f"Pod is in Failed state")
            
        if pod.status.phase == "Pending":
            issues.append(f"Pod is stuck in Pending state")
            
        if pod.status.container_statuses:
            for container in pod.status.container_statuses:
                if container.restart_count and container.restart_count > 0:
                    issues.append(f"Container {container.name} has restarted {container.restart_count} times")
                
                if not container.ready:
                    issues.append(f"Container {container.name} is not ready")
                    
        return "; ".join(issues) if issues else "Pod health degraded"


class LLMIncidentAnalyzer:
    """Uses local open-source LLMs (Llama/Ollama) or rule-based analysis (FREE VERSION)"""
    
    def __init__(self, llm_type: str = "rules", ollama_host: str = "http://localhost:11434"):
        self.llm_type = llm_type  # "rules", "ollama", "huggingface"
        self.ollama_host = ollama_host
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM client based on type
        if llm_type == "ollama":
            self._init_ollama()
        elif llm_type == "huggingface":
            self._init_huggingface()
        
        # Load knowledge base and patterns
        self.knowledge_base = self._load_knowledge_base()
        self.incident_patterns = self._load_incident_patterns()
        
        self.logger.info(f"🤖 Initialized with LLM type: {llm_type}")
    
    def _init_ollama(self):
        """Initialize Ollama client for local LLM"""
        try:
            import requests
            # Test connection to Ollama
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info("✅ Connected to Ollama server")
                models = response.json().get('models', [])
                if models:
                    self.logger.info(f"📋 Available models: {[m['name'] for m in models]}")
                else:
                    self.logger.warning("⚠️  No models found in Ollama. Run: ollama pull llama3.2")
            else:
                raise Exception("Ollama server not responding")
        except Exception as e:
            self.logger.warning(f"⚠️  Ollama not available: {e}")
            self.logger.info("📝 Fallback to rule-based analysis")
            self.llm_type = "rules"
    
    def _init_huggingface(self):
        """Initialize Hugging Face transformers for local LLM"""
        try:
            from transformers import pipeline
            self.logger.info("🔄 Loading Hugging Face model (this may take a few minutes first time)...")
            self.hf_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",  # Lightweight model for demo
                device_map="auto" if self._has_gpu() else "cpu"
            )
            self.logger.info("✅ Hugging Face model loaded")
        except ImportError:
            self.logger.warning("⚠️  transformers not installed: pip install transformers torch")
            self.llm_type = "rules"
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to load HF model: {e}")
            self.llm_type = "rules"
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load remediation knowledge base"""
        return {
            "common_issues": {
                "ImagePullBackOff": {
                    "description": "Container image cannot be pulled",
                    "common_causes": ["Invalid image name", "Registry authentication", "Network issues"],
                    "remediation": ["Check image name", "Verify registry credentials", "Check network connectivity"]
                },
                "CrashLoopBackOff": {
                    "description": "Container keeps crashing and restarting",
                    "common_causes": ["Application startup failure", "Resource limits", "Configuration issues"],
                    "remediation": ["Check application logs", "Increase resource limits", "Validate configuration"]
                },
                "OOMKilled": {
                    "description": "Container killed due to out of memory",
                    "common_causes": ["Insufficient memory limits", "Memory leak", "High load"],
                    "remediation": ["Increase memory limits", "Investigate memory usage", "Scale application"]
                }
            },
            "best_practices": [
                "Always backup before making changes",
                "Apply changes gradually",
                "Monitor impact after remediation",
                "Document all actions taken"
            ]
        }
    
    def _load_incident_patterns(self) -> Dict[str, Any]:
        """Load intelligent rule-based incident patterns (FREE VERSION)"""
        return {
            "pod_issues": {
                "ImagePullBackOff": {
                    "keywords": ["imagepullbackoff", "pull", "image", "registry"],
                    "log_patterns": ["pull access denied", "repository does not exist", "manifest unknown"],
                    "action": RemediationAction.NO_ACTION,  # Usually requires manual fix
                    "confidence": 0.9,
                    "reasoning": "Image pull failure - likely configuration issue requiring manual intervention"
                },
                "CrashLoopBackOff": {
                    "keywords": ["crashloopbackoff", "crash", "exit", "failed"],
                    "log_patterns": ["exit code", "panic", "error", "exception", "segmentation fault"],
                    "action": RemediationAction.RESTART_POD,
                    "confidence": 0.8,
                    "reasoning": "Application crash - restart may resolve transient issues"
                },
                "OOMKilled": {
                    "keywords": ["oomkilled", "memory", "killed"],
                    "log_patterns": ["out of memory", "oom", "memory limit exceeded"],
                    "action": RemediationAction.SCALE_UP,
                    "confidence": 0.85,
                    "reasoning": "Out of memory - scaling up will provide more resources"
                },
                "Pending": {
                    "keywords": ["pending", "scheduling", "unschedulable"],
                    "log_patterns": ["insufficient", "no nodes", "resource quota"],
                    "action": RemediationAction.SCALE_DOWN,
                    "confidence": 0.7,
                    "reasoning": "Resource constraints - may need to reduce resource requirements"
                }
            },
            "deployment_issues": {
                "ReplicaFailure": {
                    "keywords": ["replica", "failed", "unavailable"],
                    "conditions": ["deployment.status.ready_replicas < deployment.spec.replicas"],
                    "action": RemediationAction.ROLLBACK_DEPLOYMENT,
                    "confidence": 0.75,
                    "reasoning": "Deployment not reaching desired state - rollback to previous version"
                },
                "ProgressDeadlineExceeded": {
                    "keywords": ["deadline", "timeout", "progress"],
                    "conditions": ["deployment stuck for >10 minutes"],
                    "action": RemediationAction.ROLLBACK_DEPLOYMENT,
                    "confidence": 0.8,
                    "reasoning": "Deployment timeout exceeded - rollback to stable version"
                }
            },
            "resource_patterns": {
                "high_restart_count": {
                    "condition": lambda metrics: metrics.get('restart_count', 0) > 5,
                    "action": RemediationAction.ROLLBACK_DEPLOYMENT,
                    "confidence": 0.7,
                    "reasoning": "High restart count indicates unstable application - consider rollback"
                },
                "zero_ready_replicas": {
                    "condition": lambda metrics: metrics.get('ready_replicas', 1) == 0 and metrics.get('desired_replicas', 0) > 0,
                    "action": RemediationAction.SCALE_UP,
                    "confidence": 0.75,
                    "reasoning": "No ready replicas available - scale up to restore service"
                }
            }
        }
    
    def _analyze_with_patterns(self, incident: IncidentContext) -> RemediationPlan:
        """Advanced rule-based analysis using patterns (FREE VERSION)"""
        best_match = None
        highest_confidence = 0.0
        
        # Analyze pod-specific issues
        if incident.resource_type == "Pod":
            best_match = self._match_pod_patterns(incident)
        
        # Analyze deployment issues
        elif incident.resource_type == "Deployment":
            best_match = self._match_deployment_patterns(incident)
        
        # Check resource-based patterns
        resource_match = self._match_resource_patterns(incident)
        if resource_match and resource_match['confidence'] > (best_match['confidence'] if best_match else 0):
            best_match = resource_match
        
        # Fallback to basic analysis if no patterns match
        if not best_match:
            best_match = self._basic_analysis(incident)
        
        return RemediationPlan(
            action=best_match['action'],
            target_resource=incident.resource_name,
            target_namespace=incident.namespace,
            parameters=best_match.get('parameters', {}),
            reasoning=best_match['reasoning'],
            confidence_score=best_match['confidence'],
            estimated_impact=self._estimate_impact(best_match['action'])
        )
    
    def _match_pod_patterns(self, incident: IncidentContext) -> Optional[Dict[str, Any]]:
        """Match pod-specific patterns"""
        description_lower = incident.description.lower()
        logs_text = ' '.join(incident.logs).lower()
        
        for issue_type, pattern in self.incident_patterns['pod_issues'].items():
            # Check keywords in description
            keyword_matches = sum(1 for keyword in pattern['keywords'] if keyword in description_lower)
            
            # Check log patterns
            log_matches = sum(1 for log_pattern in pattern['log_patterns'] if log_pattern in logs_text)
            
            # Calculate match score
            total_patterns = len(pattern['keywords']) + len(pattern['log_patterns'])
            match_score = (keyword_matches + log_matches) / total_patterns if total_patterns > 0 else 0
            
            if match_score > 0.3:  # Threshold for pattern match
                confidence = pattern['confidence'] * match_score
                
                return {
                    'action': pattern['action'],
                    'confidence': min(confidence, 0.95),  # Cap at 95%
                    'reasoning': f"{pattern['reasoning']} (Pattern: {issue_type}, Match: {match_score:.2f})",
                    'parameters': self._get_action_parameters(pattern['action'], incident)
                }
        
        return None
    
    def _match_deployment_patterns(self, incident: IncidentContext) -> Optional[Dict[str, Any]]:
        """Match deployment-specific patterns"""
        description_lower = incident.description.lower()
        
        for issue_type, pattern in self.incident_patterns['deployment_issues'].items():
            keyword_matches = sum(1 for keyword in pattern['keywords'] if keyword in description_lower)
            
            if keyword_matches > 0:
                return {
                    'action': pattern['action'],
                    'confidence': pattern['confidence'],
                    'reasoning': f"{pattern['reasoning']} (Deployment issue: {issue_type})",
                    'parameters': self._get_action_parameters(pattern['action'], incident)
                }
        
        return None
    
    def _match_resource_patterns(self, incident: IncidentContext) -> Optional[Dict[str, Any]]:
        """Match resource usage patterns"""
        for pattern_name, pattern in self.incident_patterns['resource_patterns'].items():
            if pattern['condition'](incident.metrics):
                return {
                    'action': pattern['action'],
                    'confidence': pattern['confidence'],
                    'reasoning': f"{pattern['reasoning']} (Resource pattern: {pattern_name})",
                    'parameters': self._get_action_parameters(pattern['action'], incident)
                }
        
        return None
    
    def _basic_analysis(self, incident: IncidentContext) -> Dict[str, Any]:
        """Basic fallback analysis"""
        if incident.severity == IncidentSeverity.HIGH:
            return {
                'action': RemediationAction.RESTART_POD,
                'confidence': 0.6,
                'reasoning': "High severity incident - attempting pod restart as basic remediation",
                'parameters': {}
            }
        elif incident.severity == IncidentSeverity.MEDIUM:
            return {
                'action': RemediationAction.NO_ACTION,
                'confidence': 0.4,
                'reasoning': "Medium severity incident - monitoring for escalation",
                'parameters': {}
            }
        else:
            return {
                'action': RemediationAction.NO_ACTION,
                'confidence': 0.8,
                'reasoning': "Low severity incident - no action required",
                'parameters': {}
            }
    
    def _get_action_parameters(self, action: RemediationAction, incident: IncidentContext) -> Dict[str, Any]:
        """Get parameters for specific actions"""
        params = {}
        
        if action == RemediationAction.SCALE_UP:
            params['scale_factor'] = 1.5  # Scale up by 50%
            params['max_replicas'] = 10
        elif action == RemediationAction.SCALE_DOWN:
            params['scale_factor'] = 0.7  # Scale down by 30%
            params['min_replicas'] = 1
        elif action == RemediationAction.ROLLBACK_DEPLOYMENT:
            params['target_revision'] = 'previous'
        
        return params
    
    def _estimate_impact(self, action: RemediationAction) -> str:
        """Estimate impact of remediation action"""
        impact_map = {
            RemediationAction.RESTART_POD: "Brief service interruption (~10-30 seconds)",
            RemediationAction.SCALE_UP: "Increased resource usage, improved availability",
            RemediationAction.SCALE_DOWN: "Reduced resource usage, potential capacity constraints", 
            RemediationAction.ROLLBACK_DEPLOYMENT: "Service interruption during rollback (~1-3 minutes)",
            RemediationAction.UPDATE_CONFIG: "Service restart required, brief downtime",
            RemediationAction.DRAIN_NODE: "Workload migration, extended service impact",
            RemediationAction.NO_ACTION: "No immediate impact, continued monitoring"
        }
        return impact_map.get(action, "Unknown impact")
    
    async def analyze_incident(self, incident: IncidentContext) -> RemediationPlan:
        """Analyze incident using local LLM or intelligent rules (FREE VERSION)"""
        
        self.logger.info(f"Analyzing incident: {incident.resource_name} using {self.llm_type}")
        
        try:
            if self.llm_type == "ollama":
                plan = await self._analyze_with_ollama(incident)
            elif self.llm_type == "huggingface":
                plan = await self._analyze_with_huggingface(incident)
            else:
                # Use advanced pattern matching
                plan = self._analyze_with_patterns(incident)
            
            self.logger.info(f"Generated remediation plan for {incident.resource_name}: {plan.action.value}")
            return plan
            
        except Exception as e:
            self.logger.error(f"Error in {self.llm_type} analysis: {e}")
            # Always fallback to rule-based analysis
            return self._analyze_with_patterns(incident)
    
    async def _analyze_with_ollama(self, incident: IncidentContext) -> RemediationPlan:
        """Analyze using local Ollama LLM"""
        try:
            import requests
            
            # Prepare context for LLM
            context = self._prepare_llm_context(incident)
            prompt = self._build_analysis_prompt(context)
            
            # Call Ollama API
            response = requests.post(f"{self.ollama_host}/api/generate", json={
                "model": "llama3.2:latest",  # or "codellama", "mistral", etc.
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '')
                
                # Parse LLM response
                return self._parse_llm_response(llm_response, incident)
            else:
                self.logger.error(f"Ollama API error: {response.status_code}")
                return self._analyze_with_patterns(incident)
                
        except Exception as e:
            self.logger.error(f"Ollama analysis failed: {e}")
            return self._analyze_with_patterns(incident)
    
    async def _analyze_with_huggingface(self, incident: IncidentContext) -> RemediationPlan:
        """Analyze using Hugging Face local model"""
        try:
            context = self._prepare_llm_context(incident)
            prompt = self._build_analysis_prompt(context)
            
            # Generate response using HF pipeline
            responses = self.hf_pipeline(
                prompt,
                max_length=512,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True
            )
            
            if responses:
                llm_response = responses[0]['generated_text']
                return self._parse_llm_response(llm_response, incident)
            else:
                return self._analyze_with_patterns(incident)
                
        except Exception as e:
            self.logger.error(f"Hugging Face analysis failed: {e}")
            return self._analyze_with_patterns(incident)
    
    def _prepare_llm_context(self, incident: IncidentContext) -> Dict[str, Any]:
        """Prepare structured context for LLM analysis"""
        return {
            "incident": {
                "timestamp": incident.timestamp.isoformat(),
                "resource": f"{incident.resource_type}/{incident.resource_name}",
                "namespace": incident.namespace,
                "severity": incident.severity.value,
                "description": incident.description
            },
            "logs": incident.logs[-20:],  # Last 20 log lines
            "events": incident.events[:10],  # Last 10 events
            "metrics": incident.metrics
        }
    
    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build analysis prompt for local LLM"""
        return f"""You are a Kubernetes expert analyzing an incident. Provide a specific remediation plan.

INCIDENT: {context['incident']['resource']} in {context['incident']['namespace']}
STATUS: {context['incident']['description']}
SEVERITY: {context['incident']['severity']}

RECENT LOGS:
{chr(10).join(context['logs'][-5:])}

METRICS:
Restart Count: {context['metrics'].get('restart_count', 0)}
Ready Replicas: {context['metrics'].get('ready_replicas', 0)}
Desired Replicas: {context['metrics'].get('desired_replicas', 0)}

Choose ONE action and respond in JSON format:
{{
  "action": "restart_pod|scale_up|scale_down|rollback_deployment|no_action",
  "confidence_score": 0.0-1.0,
  "reasoning": "explain your decision",
  "estimated_impact": "describe expected outcome"
}}

Focus on safe, minimal-impact solutions. Consider the severity and available evidence."""
    
    def _parse_llm_response(self, response_text: str, incident: IncidentContext) -> RemediationPlan:
        """Parse LLM response into RemediationPlan"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            
            if json_match:
                json_text = json_match.group()
                response_data = json.loads(json_text)
                
                return RemediationPlan(
                    action=RemediationAction(response_data.get('action', 'no_action')),
                    target_resource=incident.resource_name,
                    target_namespace=incident.namespace,
                    parameters=response_data.get('parameters', {}),
                    reasoning=f"LLM Analysis: {response_data.get('reasoning', 'Local LLM recommendation')}",
                    confidence_score=float(response_data.get('confidence_score', 0.7)),
                    estimated_impact=response_data.get('estimated_impact', 'Unknown impact')
                )
            else:
                # No valid JSON found, fallback to patterns
                self.logger.warning("No valid JSON in LLM response, falling back to patterns")
                return self._analyze_with_patterns(incident)
                
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._analyze_with_patterns(incident)
    
    def _prepare_llm_context(self, incident: IncidentContext) -> Dict[str, Any]:
        """Prepare structured context for LLM analysis"""
        return {
            "incident": {
                "timestamp": incident.timestamp.isoformat(),
                "resource": f"{incident.resource_type}/{incident.resource_name}",
                "namespace": incident.namespace,
                "severity": incident.severity.value,
                "description": incident.description
            },
            "logs": incident.logs[-20:],  # Last 20 log lines
            "events": incident.events[:10],  # Last 10 events
            "metrics": incident.metrics,
            "knowledge_base": self.knowledge_base
        }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """
You are an expert Kubernetes Site Reliability Engineer with deep knowledge of:
- Kubernetes architecture and common failure patterns
- Container orchestration and debugging
- Production incident response and remediation
- Cloud-native application patterns

Your role is to analyze Kubernetes incidents and provide specific, actionable remediation plans.

Guidelines:
1. Analyze the provided incident data thoroughly
2. Consider both immediate fixes and root cause resolution
3. Prioritize safety and minimal disruption
4. Provide confidence scores for your recommendations
5. Always explain your reasoning
6. Consider the severity and potential impact

Available remediation actions:
- restart_pod: Restart the problematic pod
- scale_up: Increase replica count
- scale_down: Decrease replica count  
- rollback_deployment: Rollback to previous version
- update_config: Update configuration
- drain_node: Drain a problematic node
- no_action: No action needed, continue monitoring

Response format should be JSON with:
- action: one of the available actions
- parameters: action-specific parameters
- reasoning: detailed explanation
- confidence_score: 0.0-1.0
- estimated_impact: description of expected impact
"""
    
    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build analysis prompt for LLM"""
        return f"""
Analyze this Kubernetes incident and provide a remediation plan:

INCIDENT DETAILS:
{json.dumps(context['incident'], indent=2)}

RECENT LOGS:
{chr(10).join(context['logs'][-10:])}

RECENT EVENTS:
{json.dumps(context['events'][:5], indent=2, default=str)}

CURRENT METRICS:
{json.dumps(context['metrics'], indent=2)}

Please analyze this incident and provide a specific remediation plan in JSON format.
Consider the severity level and provide a confidence score for your recommendation.
"""
    
    def _parse_llm_response(self, response_text: str, incident: IncidentContext) -> RemediationPlan:
        """Parse LLM response into RemediationPlan"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
                
            json_text = response_text[start_idx:end_idx]
            response_data = json.loads(json_text)
            
            return RemediationPlan(
                action=RemediationAction(response_data.get('action', 'no_action')),
                target_resource=incident.resource_name,
                target_namespace=incident.namespace,
                parameters=response_data.get('parameters', {}),
                reasoning=response_data.get('reasoning', 'LLM analysis'),
                confidence_score=float(response_data.get('confidence_score', 0.5)),
                estimated_impact=response_data.get('estimated_impact', 'Unknown impact')
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._fallback_analysis(incident)
    
    def _fallback_analysis(self, incident: IncidentContext) -> RemediationPlan:
        """Fallback rule-based analysis if LLM fails"""
        if incident.severity == IncidentSeverity.HIGH:
            if "restart" in incident.description.lower():
                return RemediationPlan(
                    action=RemediationAction.RESTART_POD,
                    target_resource=incident.resource_name,
                    target_namespace=incident.namespace,
                    parameters={},
                    reasoning="High severity incident with restart indicators - fallback rule",
                    confidence_score=0.7,
                    estimated_impact="Pod will be restarted, brief downtime expected"
                )
        
        return RemediationPlan(
            action=RemediationAction.NO_ACTION,
            target_resource=incident.resource_name,
            target_namespace=incident.namespace,
            parameters={},
            reasoning="Insufficient data for automated remediation - fallback rule",
            confidence_score=0.3,
            estimated_impact="No immediate action, continued monitoring"
        )


class RemediationEngine:
    """Executes remediation plans on Kubernetes cluster"""
    
    def __init__(self):
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
            
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.logger = logging.getLogger(__name__)
        
        # Safety limits
        self.max_concurrent_actions = 3
        self.current_actions = 0
        
    async def execute_remediation(self, plan: RemediationPlan) -> bool:
        """Execute a remediation plan"""
        if self.current_actions >= self.max_concurrent_actions:
            self.logger.warning("Too many concurrent remediations, skipping")
            return False
        
        # Safety check - don't execute low confidence plans automatically
        if plan.confidence_score < 0.6:
            self.logger.warning(
                f"Low confidence plan ({plan.confidence_score}), requires manual approval"
            )
            return False
        
        self.current_actions += 1
        
        try:
            success = False
            
            if plan.action == RemediationAction.RESTART_POD:
                success = await self._restart_pod(plan)
            elif plan.action == RemediationAction.SCALE_UP:
                success = await self._scale_deployment(plan, scale_up=True)
            elif plan.action == RemediationAction.SCALE_DOWN:
                success = await self._scale_deployment(plan, scale_up=False)
            elif plan.action == RemediationAction.ROLLBACK_DEPLOYMENT:
                success = await self._rollback_deployment(plan)
            else:
                self.logger.info(f"No action taken for plan: {plan.action.value}")
                success = True
            
            if success:
                self.logger.info(f"Successfully executed {plan.action.value} on {plan.target_resource}")
                # Wait a bit to observe impact
                await asyncio.sleep(30)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing remediation plan: {e}")
            return False
        finally:
            self.current_actions -= 1
    
    async def _restart_pod(self, plan: RemediationPlan) -> bool:
        """Restart a pod by deleting it (deployment will recreate)"""
        try:
            self.v1.delete_namespaced_pod(
                name=plan.target_resource,
                namespace=plan.target_namespace
            )
            self.logger.info(f"Deleted pod {plan.target_resource} for restart")
            return True
        except ApiException as e:
            self.logger.error(f"Error restarting pod: {e}")
            return False
    
    async def _scale_deployment(self, plan: RemediationPlan, scale_up: bool = True) -> bool:
        """Scale a deployment up or down"""
        try:
            # Find deployment that owns this pod
            deployment_name = plan.parameters.get('deployment_name')
            if not deployment_name:
                # Try to infer from pod name
                deployment_name = plan.target_resource.rsplit('-', 2)[0]
            
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=plan.target_namespace
            )
            
            current_replicas = deployment.spec.replicas or 1
            if scale_up:
                new_replicas = min(current_replicas + 1, 10)  # Safety limit
            else:
                new_replicas = max(current_replicas - 1, 1)   # Don't scale to 0
            
            # Update deployment
            deployment.spec.replicas = new_replicas
            
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=plan.target_namespace,
                body=deployment
            )
            
            self.logger.info(
                f"Scaled deployment {deployment_name} from {current_replicas} to {new_replicas}"
            )
            return True
            
        except ApiException as e:
            self.logger.error(f"Error scaling deployment: {e}")
            return False
    
    async def _rollback_deployment(self, plan: RemediationPlan) -> bool:
        """Rollback deployment to previous version"""
        try:
            deployment_name = plan.parameters.get('deployment_name', plan.target_resource)
            
            # Get rollout history
            rollout_history = self.apps_v1.list_namespaced_replica_set(
                namespace=plan.target_namespace,
                label_selector=f"app={deployment_name}"
            )
            
            if len(rollout_history.items) < 2:
                self.logger.warning("No previous version found for rollback")
                return False
            
            # Trigger rollback using kubectl (simpler than manual revision handling)
            import subprocess
            result = subprocess.run([
                'kubectl', 'rollout', 'undo', 
                f'deployment/{deployment_name}',
                '-n', plan.target_namespace
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Rollback initiated for deployment {deployment_name}")
                return True
            else:
                self.logger.error(f"Rollback failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error rolling back deployment: {e}")
            return False


class IncidentResponseEngine:
    """Main orchestrator for incident response (FREE + LOCAL LLM VERSION)"""
    
    def __init__(self, llm_type: str = "rules", ollama_host: str = "http://localhost:11434"):
        self.monitor = KubernetesMonitor()
        self.analyzer = LLMIncidentAnalyzer(llm_type=llm_type, ollama_host=ollama_host)
        self.remediation_engine = RemediationEngine()
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.scan_interval = 60  # seconds
        self.incident_history = []
        
        if llm_type == "rules":
            self.logger.info("🆓 FREE VERSION: Using intelligent rule-based analysis")
        else:
            self.logger.info(f"🤖 LOCAL LLM VERSION: Using {llm_type} for analysis")
        
    async def run_continuous_monitoring(self):
        """Run continuous incident monitoring and response"""
        self.logger.info("Starting continuous incident monitoring")
        
        while True:
            try:
                # Detect incidents
                incidents = self.monitor.detect_pod_issues()
                
                if incidents:
                    self.logger.info(f"Detected {len(incidents)} incidents")
                    
                    # Process each incident
                    for incident in incidents:
                        await self._handle_incident(incident)
                        
                        # Rate limiting - don't overwhelm the cluster
                        await asyncio.sleep(5)
                        
                else:
                    self.logger.debug("No incidents detected")
                
                # Wait before next scan
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _handle_incident(self, incident: IncidentContext):
        """Handle a single incident"""
        self.logger.info(
            f"Handling {incident.severity.value} incident: "
            f"{incident.resource_type}/{incident.resource_name} in {incident.namespace}"
        )
        
        # Check if we've already handled this incident recently
        if self._is_duplicate_incident(incident):
            self.logger.info("Duplicate incident, skipping")
            return
        
        # Analyze incident and generate remediation plan
        plan = await self.analyzer.analyze_incident(incident)
        
        self.logger.info(
            f"Generated plan: {plan.action.value} "
            f"(confidence: {plan.confidence_score:.2f})"
        )
        
        # Execute remediation if confidence is high enough
        if plan.action != RemediationAction.NO_ACTION:
            success = await self.remediation_engine.execute_remediation(plan)
            
            # Log the outcome
            outcome = "SUCCESS" if success else "FAILED"
            self.logger.info(f"Remediation {outcome}: {plan.reasoning}")
            
        # Record incident in history
        self.incident_history.append({
            'incident': incident,
            'plan': plan,
            'timestamp': datetime.now()
        })
        
        # Keep history manageable
        if len(self.incident_history) > 1000:
            self.incident_history = self.incident_history[-500:]
    
    def _is_duplicate_incident(self, incident: IncidentContext) -> bool:
        """Check if we've handled this incident recently"""
        cutoff = datetime.now() - timedelta(minutes=10)
        
        for historical in self.incident_history:
            if (historical['timestamp'] > cutoff and
                historical['incident'].resource_name == incident.resource_name and
                historical['incident'].namespace == incident.namespace):
                return True
                
        return False
    
    def get_incident_summary(self) -> Dict[str, Any]:
        """Get summary of recent incidents and actions"""
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_incidents = [
            h for h in self.incident_history 
            if h['timestamp'] > recent_cutoff
        ]
        
        summary = {
            'total_incidents_24h': len(recent_incidents),
            'incidents_by_severity': {},
            'actions_taken': {},
            'top_affected_resources': {}
        }
        
        for incident_data in recent_incidents:
            incident = incident_data['incident']
            plan = incident_data['plan']
            
            # Count by severity
            severity = incident.severity.value
            summary['incidents_by_severity'][severity] = \
                summary['incidents_by_severity'].get(severity, 0) + 1
            
            # Count actions taken
            action = plan.action.value
            summary['actions_taken'][action] = \
                summary['actions_taken'].get(action, 0) + 1
            
            # Track affected resources
            resource_key = f"{incident.namespace}/{incident.resource_name}"
            summary['top_affected_resources'][resource_key] = \
                summary['top_affected_resources'].get(resource_key, 0) + 1
        
        return summary


# Main execution
async def main():
    """Main entry point (FREE + LOCAL LLM VERSION)"""
    
    # Check for LLM preference from environment
    llm_type = os.getenv('LLM_TYPE', 'rules')  # rules, ollama, huggingface
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    
    print("🤖 LLM-Powered Kubernetes Incident Response System")
    
    if llm_type == "ollama":
        print("🦙 Using Ollama (Local Llama/Mistral/etc.) for advanced analysis")
        print("📝 Make sure Ollama is running: ollama serve")
    elif llm_type == "huggingface": 
        print("🤗 Using Hugging Face transformers for local LLM analysis")
        print("📝 First run may take time to download models")
    else:
        print("🆓 Using intelligent rule-based analysis (completely free)")
    
    print("💡 All analysis runs locally - no external API costs!")
    
    # Initialize and start the incident response engine
    engine = IncidentResponseEngine(llm_type=llm_type, ollama_host=ollama_host)
    
    print("🚀 Starting Kubernetes incident response system...")
    print("🔍 Monitoring cluster for incidents and ready to auto-remediate...")
    
    try:
        await engine.run_continuous_monitoring()
    except KeyboardInterrupt:
        print("\n👋 Shutting down incident response system")
        
        # Print final summary
        summary = engine.get_incident_summary()
        print("\n📊 Final Summary:")
        print(f"Total incidents in last 24h: {summary['total_incidents_24h']}")
        print(f"Actions taken: {summary['actions_taken']}")
        print(f"🤖 All operations completed using {llm_type} analysis!")


if __name__ == "__main__":
    asyncio.run(main())
