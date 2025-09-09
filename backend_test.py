import requests
import sys
import json
import base64
from datetime import datetime

class TruthLensAPITester:
    def __init__(self, base_url="https://truth-scout.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session_token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.session = requests.Session()

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, cookies=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}
        
        if cookies:
            self.session.cookies.update(cookies)

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = self.session.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    # Remove Content-Type for file uploads
                    headers.pop('Content-Type', None)
                    response = self.session.post(url, data=data, files=files, headers=headers)
                else:
                    response = self.session.post(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = self.session.delete(url, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        return self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )

    def test_analyze_text(self):
        """Test text analysis endpoint"""
        test_text = "Breaking: Scientists discover that drinking water is actually dangerous! Share to save lives!"
        
        success, response = self.run_test(
            "Text Analysis",
            "POST",
            "analyze",
            200,
            data={
                "text": test_text,
                "content_type": "text"
            }
        )
        
        if success and response:
            # Validate response structure
            required_fields = ['id', 'content', 'credibility_score', 'verdict', 'confidence', 'reasoning', 'education_tips']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Missing fields in response: {missing_fields}")
                return False
            
            # Validate data types and ranges
            if not (0 <= response.get('credibility_score', -1) <= 100):
                print(f"⚠️  Invalid credibility_score: {response.get('credibility_score')}")
                return False
                
            if not (0.0 <= response.get('confidence', -1) <= 1.0):
                print(f"⚠️  Invalid confidence: {response.get('confidence')}")
                return False
                
            print(f"   Credibility Score: {response.get('credibility_score')}/100")
            print(f"   Verdict: {response.get('verdict')}")
            print(f"   Confidence: {response.get('confidence')}")
            
        return success

    def test_analyze_image(self):
        """Test image analysis endpoint with a simple base64 image"""
        # Create a simple 1x1 pixel PNG image in base64
        simple_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        success, response = self.run_test(
            "Image Analysis",
            "POST",
            "analyze",
            200,
            data={
                "image_base64": simple_png_base64,
                "content_type": "image",
                "text": "Test image for misinformation analysis"
            }
        )
        
        if success and response:
            print(f"   Image Analysis - Credibility Score: {response.get('credibility_score')}/100")
            print(f"   Verdict: {response.get('verdict')}")
            
        return success

    def test_analyze_invalid_request(self):
        """Test analysis with invalid request (no content)"""
        return self.run_test(
            "Invalid Analysis Request",
            "POST",
            "analyze",
            400,
            data={}
        )

    def test_public_history(self):
        """Test public history endpoint"""
        return self.run_test(
            "Public History",
            "GET",
            "public-history",
            200
        )

    def test_auth_me_unauthenticated(self):
        """Test /auth/me without authentication"""
        return self.run_test(
            "Auth Me (Unauthenticated)",
            "GET",
            "auth/me",
            401
        )

    def test_history_unauthenticated(self):
        """Test /history without authentication"""
        return self.run_test(
            "History (Unauthenticated)",
            "GET",
            "history",
            401
        )

    def test_logout_unauthenticated(self):
        """Test logout without authentication"""
        return self.run_test(
            "Logout (Unauthenticated)",
            "POST",
            "auth/logout",
            401
        )

    def test_session_creation_invalid(self):
        """Test session creation with invalid data"""
        return self.run_test(
            "Session Creation (Invalid)",
            "POST",
            "auth/session",
            400,
            data={}
        )

def main():
    print("🚀 Starting TruthLens API Testing...")
    print("=" * 50)
    
    # Setup
    tester = TruthLensAPITester()
    
    # Run basic API tests
    print("\n📋 Basic API Tests")
    print("-" * 30)
    tester.test_root_endpoint()
    
    # Test analysis endpoints
    print("\n🔍 Analysis Tests")
    print("-" * 30)
    tester.test_analyze_text()
    tester.test_analyze_image()
    tester.test_analyze_invalid_request()
    
    # Test history endpoints
    print("\n📚 History Tests")
    print("-" * 30)
    tester.test_public_history()
    tester.test_history_unauthenticated()
    
    # Test authentication endpoints
    print("\n🔐 Authentication Tests")
    print("-" * 30)
    tester.test_auth_me_unauthenticated()
    tester.test_session_creation_invalid()
    tester.test_logout_unauthenticated()
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        failed_tests = tester.tests_run - tester.tests_passed
        print(f"⚠️  {failed_tests} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())