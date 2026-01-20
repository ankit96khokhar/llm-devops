#!/usr/bin/env python3
"""
ArgoCD Integration for LLM Incident Response

This module provides integration with ArgoCD for automated rollbacks
and deployment management during incident response.
"""

import asyncio
import json
import logging
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import base64


@dataclass
class ArgoCDConfig:
    """ArgoCD configuration"""
    server_url: str
    username: str
    password: str
    token: Optional[str] = None
    verify_ssl: bool = True


@dataclass
class ApplicationStatus:
    """ArgoCD application status"""
    name: str
    namespace: str
    sync_status: str
    health_status: str
    revision: str
    last_sync: Optional[datetime] = None


class ArgoCDClient:
    """Client for interacting with ArgoCD API"""
    
    def __init__(self, config: ArgoCDConfig):
        self.config = config
        self.session = requests.Session()
        self.session.verify = config.verify_ssl
        self.logger = logging.getLogger(__name__)
        
        # Authenticate on initialization
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with ArgoCD and get session token"""
        try:
            if self.config.token:
                self.session.headers.update({
                    'Authorization': f'Bearer {self.config.token}'
                })
            else:
                # Login with username/password
                auth_url = f"{self.config.server_url}/api/v1/session"
                auth_data = {
                    "username": self.config.username,
                    "password": self.config.password
                }
                
                response = self.session.post(auth_url, json=auth_data)
                response.raise_for_status()
                
                token = response.json().get('token')
                if token:
                    self.session.headers.update({
                        'Authorization': f'Bearer {token}'
                    })
                    self.logger.info("Successfully authenticated with ArgoCD")
                else:
                    raise Exception("No token received from ArgoCD")
                    
        except Exception as e:
            self.logger.error(f"Failed to authenticate with ArgoCD: {e}")
            raise
    
    def get_applications(self) -> List[ApplicationStatus]:
        """Get all ArgoCD applications"""
        try:
            url = f"{self.config.server_url}/api/v1/applications"
            response = self.session.get(url)
            response.raise_for_status()
            
            applications = []
            for app_data in response.json().get('items', []):
                app = ApplicationStatus(
                    name=app_data['metadata']['name'],
                    namespace=app_data['metadata']['namespace'],
                    sync_status=app_data['status']['sync']['status'],
                    health_status=app_data['status']['health']['status'],
                    revision=app_data['status']['sync']['revision'],
                    last_sync=self._parse_timestamp(
                        app_data['status'].get('operationState', {}).get('finishedAt')
                    )
                )
                applications.append(app)
            
            return applications
            
        except Exception as e:
            self.logger.error(f"Error getting applications: {e}")
            return []
    
    def get_application(self, app_name: str) -> Optional[ApplicationStatus]:
        """Get specific ArgoCD application"""
        try:
            url = f"{self.config.server_url}/api/v1/applications/{app_name}"
            response = self.session.get(url)
            response.raise_for_status()
            
            app_data = response.json()
            return ApplicationStatus(
                name=app_data['metadata']['name'],
                namespace=app_data['metadata']['namespace'],
                sync_status=app_data['status']['sync']['status'],
                health_status=app_data['status']['health']['status'],
                revision=app_data['status']['sync']['revision'],
                last_sync=self._parse_timestamp(
                    app_data['status'].get('operationState', {}).get('finishedAt')
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error getting application {app_name}: {e}")
            return None
    
    def rollback_application(self, app_name: str, revision: Optional[str] = None) -> bool:
        """Rollback application to previous or specific revision"""
        try:
            if not revision:
                # Get previous revision from history
                history = self.get_application_history(app_name)
                if len(history) < 2:
                    self.logger.error(f"No previous revision found for {app_name}")
                    return False
                revision = history[-2]['revision']  # Get second-to-last revision
            
            # Sync to specific revision
            url = f"{self.config.server_url}/api/v1/applications/{app_name}/sync"
            sync_data = {
                "revision": revision,
                "prune": True,
                "dryRun": False,
                "strategy": {
                    "apply": {
                        "force": False
                    }
                }
            }
            
            response = self.session.post(url, json=sync_data)
            response.raise_for_status()
            
            self.logger.info(f"Successfully initiated rollback of {app_name} to revision {revision}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rolling back application {app_name}: {e}")
            return False
    
    def sync_application(self, app_name: str, prune: bool = True) -> bool:
        """Sync application to desired state"""
        try:
            url = f"{self.config.server_url}/api/v1/applications/{app_name}/sync"
            sync_data = {
                "prune": prune,
                "dryRun": False,
                "strategy": {
                    "apply": {
                        "force": False
                    }
                }
            }
            
            response = self.session.post(url, json=sync_data)
            response.raise_for_status()
            
            self.logger.info(f"Successfully initiated sync of {app_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing application {app_name}: {e}")
            return False
    
    def get_application_history(self, app_name: str) -> List[Dict[str, Any]]:
        """Get application deployment history"""
        try:
            url = f"{self.config.server_url}/api/v1/applications/{app_name}/revisions"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting history for {app_name}: {e}")
            return []
    
    def get_unhealthy_applications(self) -> List[ApplicationStatus]:
        """Get applications that are not healthy"""
        applications = self.get_applications()
        return [
            app for app in applications 
            if app.health_status not in ['Healthy', 'Progressing']
        ]
    
    def get_out_of_sync_applications(self) -> List[ApplicationStatus]:
        """Get applications that are out of sync"""
        applications = self.get_applications()
        return [
            app for app in applications 
            if app.sync_status != 'Synced'
        ]
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse ArgoCD timestamp format"""
        if not timestamp_str:
            return None
        
        try:
            # ArgoCD uses RFC3339 format
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            return None


class ArgoCDIncidentHandler:
    """Handles ArgoCD-related incidents and integrates with main incident response"""
    
    def __init__(self, argocd_config: ArgoCDConfig):
        self.client = ArgoCDClient(argocd_config)
        self.logger = logging.getLogger(__name__)
        
    async def check_argocd_health(self) -> List[Dict[str, Any]]:
        """Check for ArgoCD-related issues"""
        incidents = []
        
        # Check for unhealthy applications
        unhealthy_apps = self.client.get_unhealthy_applications()
        for app in unhealthy_apps:
            incidents.append({
                'type': 'argocd_unhealthy_app',
                'severity': 'high' if app.health_status == 'Degraded' else 'medium',
                'resource': f'argocd-app/{app.name}',
                'namespace': app.namespace,
                'description': f'Application {app.name} is {app.health_status}',
                'metadata': {
                    'app_name': app.name,
                    'health_status': app.health_status,
                    'sync_status': app.sync_status,
                    'revision': app.revision
                }
            })
        
        # Check for out-of-sync applications
        out_of_sync_apps = self.client.get_out_of_sync_applications()
        for app in out_of_sync_apps:
            # Only report if out of sync for more than 5 minutes
            if app.last_sync and datetime.now() - app.last_sync > timedelta(minutes=5):
                incidents.append({
                    'type': 'argocd_out_of_sync',
                    'severity': 'medium',
                    'resource': f'argocd-app/{app.name}',
                    'namespace': app.namespace,
                    'description': f'Application {app.name} is out of sync',
                    'metadata': {
                        'app_name': app.name,
                        'sync_status': app.sync_status,
                        'revision': app.revision,
                        'last_sync': app.last_sync
                    }
                })
        
        return incidents
    
    async def handle_deployment_incident(self, namespace: str, deployment_name: str) -> bool:
        """Handle incident by finding related ArgoCD app and taking action"""
        try:
            # Find ArgoCD application responsible for this deployment
            applications = self.client.get_applications()
            
            target_app = None
            for app in applications:
                if app.namespace == namespace:
                    # Check if this app manages the deployment
                    # This is a simplified check - in reality you might need more sophisticated matching
                    target_app = app
                    break
            
            if not target_app:
                self.logger.info(f"No ArgoCD application found for deployment {deployment_name} in {namespace}")
                return False
            
            self.logger.info(f"Found ArgoCD application {target_app.name} for deployment {deployment_name}")
            
            # Check if app is healthy
            if target_app.health_status not in ['Healthy']:
                self.logger.info(f"ArgoCD app {target_app.name} is {target_app.health_status}, attempting rollback")
                success = self.client.rollback_application(target_app.name)
                
                if success:
                    self.logger.info(f"Initiated rollback for ArgoCD application {target_app.name}")
                    return True
                else:
                    self.logger.error(f"Failed to rollback ArgoCD application {target_app.name}")
            
            # If app is healthy but deployment is having issues, try re-sync
            elif target_app.sync_status != 'Synced':
                self.logger.info(f"Re-syncing ArgoCD application {target_app.name}")
                success = self.client.sync_application(target_app.name)
                
                if success:
                    self.logger.info(f"Initiated sync for ArgoCD application {target_app.name}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling deployment incident via ArgoCD: {e}")
            return False
    
    def get_application_insights(self, app_name: str) -> Dict[str, Any]:
        """Get detailed insights about an application for LLM analysis"""
        app = self.client.get_application(app_name)
        if not app:
            return {}
        
        history = self.client.get_application_history(app_name)
        
        return {
            'current_status': {
                'health': app.health_status,
                'sync': app.sync_status,
                'revision': app.revision,
                'last_sync': app.last_sync.isoformat() if app.last_sync else None
            },
            'deployment_history': history[-5:],  # Last 5 deployments
            'recommendations': self._generate_recommendations(app, history)
        }
    
    def _generate_recommendations(self, app: ApplicationStatus, history: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on app status and history"""
        recommendations = []
        
        if app.health_status == 'Degraded':
            recommendations.append("Application is degraded - consider rollback to previous version")
        
        if app.sync_status == 'OutOfSync':
            recommendations.append("Application is out of sync - sync to desired state")
        
        # Check deployment frequency
        if len(history) > 1:
            recent_deployments = [
                h for h in history 
                if self._parse_timestamp_dict(h.get('deployedAt')) and 
                datetime.now() - self._parse_timestamp_dict(h.get('deployedAt')) < timedelta(hours=1)
            ]
            
            if len(recent_deployments) > 3:
                recommendations.append("High deployment frequency detected - investigate deployment pipeline")
        
        return recommendations
    
    def _parse_timestamp_dict(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp from dictionary"""
        if not timestamp_str:
            return None
        
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            return None


# Configuration helper
def create_argocd_config_from_env() -> ArgoCDConfig:
    """Create ArgoCD config from environment variables"""
    return ArgoCDConfig(
        server_url=os.getenv('ARGOCD_SERVER', 'https://argocd.default.svc.cluster.local'),
        username=os.getenv('ARGOCD_USERNAME', 'admin'),
        password=os.getenv('ARGOCD_PASSWORD', ''),
        token=os.getenv('ARGOCD_TOKEN'),
        verify_ssl=os.getenv('ARGOCD_VERIFY_SSL', 'true').lower() == 'true'
    )


# Example usage
async def main():
    """Example usage of ArgoCD integration"""
    try:
        # Create config from environment
        config = create_argocd_config_from_env()
        
        # Initialize handler
        handler = ArgoCDIncidentHandler(config)
        
        # Check for ArgoCD-related incidents
        incidents = await handler.check_argocd_health()
        
        if incidents:
            print(f"Found {len(incidents)} ArgoCD-related incidents:")
            for incident in incidents:
                print(f"- {incident['resource']}: {incident['description']}")
        else:
            print("No ArgoCD incidents detected")
            
        # Get application insights
        apps = handler.client.get_applications()
        for app in apps[:3]:  # Show first 3 apps
            insights = handler.get_application_insights(app.name)
            print(f"\nInsights for {app.name}:")
            print(f"  Status: {insights.get('current_status', {})}")
            print(f"  Recommendations: {insights.get('recommendations', [])}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
