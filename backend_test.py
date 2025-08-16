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

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None):
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
        """Test /api/health endpoint"""
        success, response = self.run_test("Health Check", "GET", "health", 200)
        if success:
            if response.get('status') == 'ok':
                self.log("   ✓ Status is 'ok'")
            else:
                self.log(f"   ⚠️  Status is '{response.get('status')}', expected 'ok'")
            
            mongo_status = response.get('mongo')
            if mongo_status == 'ok':
                self.log("   ✓ MongoDB is healthy")
            else:
                self.log(f"   ⚠️  MongoDB status: '{mongo_status}'")
        return success

    def test_version(self):
        """Test /api/version endpoint"""
        success, response = self.run_test("Version Check", "GET", "version", 200)
        if success:
            version = response.get('version')
            if version:
                self.log(f"   ✓ Version: {version}")
            else:
                self.log("   ⚠️  No version in response")
        return success

    def test_create_status(self, client_name="tester"):
        """Test POST /api/status endpoint"""
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

    def test_list_status(self, limit=5, offset=0, q="test"):
        """Test GET /api/status endpoint with pagination and search"""
        params = {"limit": limit, "offset": offset}
        if q:
            params["q"] = q
            
        success, response = self.run_test(
            f"List Status (limit={limit}, offset={offset}, q='{q}')", 
            "GET", 
            "status", 
            200,
            params=params
        )
        if success:
            if isinstance(response, list):
                self.log(f"   ✓ Returned {len(response)} items (max {limit})")
                if len(response) <= limit:
                    self.log("   ✓ Respects limit parameter")
                else:
                    self.log(f"   ⚠️  Returned more items than limit: {len(response)} > {limit}")
                    
                # Check if search filter works
                if q and response:
                    filtered_correctly = all(q.lower() in item.get('client_name', '').lower() for item in response)
                    if filtered_correctly:
                        self.log(f"   ✓ Search filter working correctly for '{q}'")
                    else:
                        self.log(f"   ⚠️  Search filter may not be working for '{q}'")
            else:
                self.log(f"   ⚠️  Expected array, got: {type(response)}")
        return success

    def test_status_count(self):
        """Test GET /api/status/count endpoint"""
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

    def run_all_tests(self):
        """Run all backend API tests"""
        self.log("🚀 Starting Backend API Tests")
        self.log("=" * 50)
        
        # Test basic endpoints
        self.test_health()
        self.test_version()
        
        # Test status creation
        test_client = f"tester-{int(time.time())}"
        status_id = self.test_create_status(test_client)
        
        # Test listing with different parameters
        self.test_list_status(limit=5, offset=0, q="test")
        self.test_list_status(limit=10, offset=0, q="")  # No search filter
        
        # Test count endpoint
        self.test_status_count()
        
        # Summary
        self.log("=" * 50)
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