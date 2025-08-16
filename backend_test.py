#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime
import time

class BackendAPITester:
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.created_status_ids = []

    def log(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None, expect_text=False):
        """Run a single API test"""
        url = f"{self.base_url}/api/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        self.log(f"🔍 Testing {name}...")
        self.log(f"   URL: {method} {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status
            
            if success:
                self.tests_passed += 1
                self.log(f"✅ PASSED - Status: {response.status_code}")
                if expect_text:
                    self.log(f"   Response (text): {response.text[:200]}...")
                    return True, response.text
                else:
                    try:
                        response_data = response.json()
                        self.log(f"   Response: {json.dumps(response_data, indent=2, default=str)}")
                        return True, response_data
                    except:
                        self.log(f"   Response: {response.text}")
                        return True, {}
            else:
                self.log(f"❌ FAILED - Expected {expected_status}, got {response.status_code}")
                self.log(f"   Response: {response.text}")
                return False, {}

        except Exception as e:
            self.log(f"❌ FAILED - Error: {str(e)}")
            return False, {}

    def test_health(self):
        """Test 1: GET /api/health -> 200, contains {status:"ok"}"""
        success, response = self.run_test("Health Check", "GET", "health", 200)
        if success:
            if response.get('status') == 'ok':
                self.log("   ✓ Status is 'ok'")
            else:
                self.log(f"   ⚠️  Status is '{response.get('status')}', expected 'ok'")
        return success

    def test_version(self):
        """Test 2: GET /api/version -> 200, version string"""
        success, response = self.run_test("Version Check", "GET", "version", 200)
        if success:
            version = response.get('version')
            if version:
                self.log(f"   ✓ Version: {version}")
            else:
                self.log("   ⚠️  No version in response")
        return success

    def test_metrics(self):
        """Test 3: GET /api/metrics -> 200, text/plain Prometheus exposition beginning with # HELP"""
        success, response = self.run_test("Metrics Check", "GET", "metrics", 200, expect_text=True)
        if success:
            if response.startswith('# HELP'):
                self.log("   ✓ Prometheus metrics format detected")
            else:
                self.log(f"   ⚠️  Response doesn't start with '# HELP': {response[:50]}...")
        return success

    def test_create_status(self, client_name="e2e-tester"):
        """Test 4: POST /api/status {client_name:"e2e-tester"} -> 200 with id/client_name/timestamp"""
        success, response = self.run_test(
            "Create Status", 
            "POST", 
            "status", 
            200, 
            data={"client_name": client_name}
        )
        if success:
            status_id = response.get('id')
            returned_client = response.get('client_name')
            timestamp = response.get('timestamp')
            
            if status_id:
                self.log(f"   ✓ Created status with ID: {status_id}")
                self.created_status_ids.append(status_id)
            else:
                self.log("   ⚠️  No ID in response")
                
            if returned_client == client_name:
                self.log(f"   ✓ Client name matches: {returned_client}")
            else:
                self.log(f"   ⚠️  Client name mismatch: expected '{client_name}', got '{returned_client}'")
                
            if timestamp:
                self.log(f"   ✓ Timestamp: {timestamp}")
            else:
                self.log("   ⚠️  No timestamp in response")
                
            return status_id if success else None
        return None

    def test_status_count(self):
        """Test 5: GET /api/status/count -> 200 with {total, distinct_clients}"""
        success, response = self.run_test("Status Count", "GET", "status/count", 200)
        if success:
            total = response.get('total')
            distinct_clients = response.get('distinct_clients')
            
            if isinstance(total, int) and total >= 0:
                self.log(f"   ✓ Total count: {total}")
            else:
                self.log(f"   ⚠️  Invalid total count: {total}")
                
            if isinstance(distinct_clients, int) and distinct_clients >= 0:
                self.log(f"   ✓ Distinct clients: {distinct_clients}")
            else:
                self.log(f"   ⚠️  Invalid distinct clients count: {distinct_clients}")
        return success

    def test_export_csv(self, client="e2e-tester"):
        """Test 6: GET /api/status/export?client=e2e-tester -> 200 CSV with header line"""
        success, response = self.run_test(
            "Export CSV", 
            "GET", 
            "status/export", 
            200, 
            params={"client": client},
            expect_text=True
        )
        if success:
            lines = response.split('\n')
            if lines and 'id,client_name,timestamp' in lines[0]:
                self.log("   ✓ CSV header detected")
            else:
                self.log(f"   ⚠️  Expected CSV header, got: {lines[0] if lines else 'empty'}")
        return success

    def test_purge_status(self, client_name="e2e-tester", older_than_hours=1):
        """Test 7: POST /api/status/purge with body {client_name:"e2e-tester", older_than_hours:1} -> 200 {deleted: number}"""
        success, response = self.run_test(
            "Purge Status", 
            "POST", 
            "status/purge", 
            200, 
            data={"client_name": client_name, "older_than_hours": older_than_hours}
        )
        if success:
            deleted = response.get('deleted')
            if isinstance(deleted, int) and deleted >= 0:
                self.log(f"   ✓ Deleted count: {deleted}")
            else:
                self.log(f"   ⚠️  Invalid deleted count: {deleted}")
        return success

    def test_config(self):
        """Test 8: GET /api/config -> 200 {admin_enabled:true}"""
        success, response = self.run_test("Config Check", "GET", "config", 200)
        if success:
            admin_enabled = response.get('admin_enabled')
            if admin_enabled is True:
                self.log("   ✓ Admin enabled: true")
            else:
                self.log(f"   ⚠️  Admin enabled: {admin_enabled}")
        return success

    def test_providers_status(self):
        """Test 9: GET /api/providers/status -> 200 with booleans for local/together/openrouter"""
        success, response = self.run_test("Providers Status", "GET", "providers/status", 200)
        if success:
            local = response.get('local')
            together = response.get('together')
            openrouter = response.get('openrouter')
            
            self.log(f"   ✓ Local provider: {local}")
            self.log(f"   ✓ Together provider: {together}")
            self.log(f"   ✓ OpenRouter provider: {openrouter}")
        return success

    def test_ai_chat(self, provider="local", model="llama3"):
        """Test 10: POST /api/ai/chat with simple prompt -> Expect 200 if available; else 503"""
        data = {
            "messages": [{"role": "user", "content": "Hello, respond with just 'Hi'"}],
            "model": model,
            "provider": provider,
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        success, response = self.run_test(
            f"AI Chat ({provider})", 
            "POST", 
            "ai/chat", 
            200,  # We'll accept both 200 and 503
            data=data
        )
        
        # If we get 503, that's also acceptable per the requirements
        if not success:
            # Try again expecting 503
            self.tests_run -= 1  # Don't double count
            success_503, response_503 = self.run_test(
                f"AI Chat ({provider}) - Expected 503", 
                "POST", 
                "ai/chat", 
                503,
                data=data
            )
            if success_503:
                detail = response_503.get('detail', '')
                if 'No LLM providers available or all failed' in detail:
                    self.log("   ✓ Expected 503 error for unavailable providers")
                    return True
        
        if success:
            content = response.get('content')
            provider_used = response.get('provider')
            model_used = response.get('model')
            
            if content:
                self.log(f"   ✓ AI response: {content}")
            if provider_used:
                self.log(f"   ✓ Provider used: {provider_used}")
            if model_used:
                self.log(f"   ✓ Model used: {model_used}")
        
        return success

    def run_all_tests(self):
        """Run all backend API tests as specified in the review request"""
        self.log("🚀 Starting Backend API Tests")
        self.log("=" * 60)
        
        # Test 1: Health
        self.test_health()
        
        # Test 2: Version
        self.test_version()
        
        # Test 3: Metrics
        self.test_metrics()
        
        # Test 4: Create Status
        test_client = "e2e-tester"
        status_id = self.test_create_status(test_client)
        
        # Test 5: Status Count
        self.test_status_count()
        
        # Test 6: Export CSV
        self.test_export_csv(test_client)
        
        # Test 7: Purge Status
        self.test_purge_status(test_client, 1)
        
        # Test 8: Config
        self.test_config()
        
        # Test 9: Providers Status
        self.test_providers_status()
        
        # Test 10: AI Chat endpoints
        self.test_ai_chat("local", "llama3")
        self.test_ai_chat("together", "meta-llama/Llama-3-8B-Instruct-Turbo")
        self.test_ai_chat("openrouter", "meta-llama/llama-3-8b-instruct")
        
        # Summary
        self.log("=" * 60)
        self.log(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        
        if self.tests_passed == self.tests_run:
            self.log("🎉 All tests passed!")
            return 0
        else:
            self.log("❌ Some tests failed!")
            return 1

def main():
    """Main test runner"""
    # Test against the internal backend port
    tester = BackendAPITester("http://127.0.0.1:8001")
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())