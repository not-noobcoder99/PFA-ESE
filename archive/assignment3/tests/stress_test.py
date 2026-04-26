"""
Stress testing suite for API performance and robustness analysis
Tests latency, throughput, error handling, and edge cases
"""

import time
import json
import statistics
from typing import List, Dict, Any
import requests
from datetime import datetime
import random


class APIStressTester:
    """Comprehensive API stress and robustness testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "single_prediction": [],
            "batch_predictions": [],
            "error_cases": [],
            "concurrent_requests": []
        }
    
    def generate_patient(self, seed: int = None) -> Dict[str, Any]:
        """Generate random but clinically valid patient record"""
        if seed is not None:
            random.seed(seed)
        
        return {
            "age": random.randint(30, 80),
            "sex": random.randint(0, 1),
            "cp": random.randint(0, 3),
            "trestbps": random.randint(90, 200),
            "chol": random.randint(130, 560),
            "fbs": random.randint(0, 1),
            "restecg": random.randint(0, 2),
            "thalach": random.randint(60, 200),
            "exang": random.randint(0, 1),
            "oldpeak": round(random.uniform(0, 6), 1),
            "slope": random.randint(0, 2),
            "ca": random.randint(0, 4),
            "thal": random.randint(0, 3)
        }
    
    def test_single_prediction_latency(self, num_requests: int = 100) -> Dict[str, float]:
        """Test latency for single predictions"""
        print(f"\n[TEST] Single Prediction Latency ({num_requests} requests)...")
        
        latencies = []
        
        for i in range(num_requests):
            patient = self.generate_patient(seed=i)
            
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json=patient,
                    timeout=5
                )
                latency = (time.time() - start) * 1000  # ms
                
                if response.status_code == 200:
                    latencies.append(latency)
                    if (i + 1) % 20 == 0:
                        print(f"  ✓ Completed {i + 1}/{num_requests}")
            except Exception as e:
                print(f"  ✗ Request {i + 1} failed: {str(e)}")
        
        stats = {
            "num_requests": len(latencies),
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        }
        
        print(f"  Mean Latency: {stats['mean_latency_ms']:.2f}ms")
        print(f"  P95 Latency: {stats['p95_latency_ms']:.2f}ms")
        print(f"  P99 Latency: {stats['p99_latency_ms']:.2f}ms")
        
        self.results["single_prediction"].append(stats)
        return stats
    
    def test_batch_throughput(self, batch_sizes: List[int] = [10, 25, 50, 100]) -> Dict[int, float]:
        """Test throughput with different batch sizes"""
        print(f"\n[TEST] Batch Throughput ({len(batch_sizes)} batch sizes)...")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            batch_data = {
                "patients": [self.generate_patient(seed=i) for i in range(batch_size)]
            }
            
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/api/predict/batch",
                    json=batch_data,
                    timeout=30
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    throughput = batch_size / elapsed  # predictions/sec
                    batch_results[batch_size] = {
                        "time_seconds": elapsed,
                        "throughput_predictions_per_sec": throughput,
                        "avg_latency_ms": data.get("average_latency_ms", 0)
                    }
                    print(f"  ✓ Batch size {batch_size}: {throughput:.2f} pred/sec")
            except Exception as e:
                print(f"  ✗ Batch {batch_size} failed: {str(e)}")
        
        self.results["batch_predictions"].append(batch_results)
        return batch_results
    
    def test_error_handling(self) -> Dict[str, int]:
        """Test robustness to invalid inputs"""
        print(f"\n[TEST] Error Handling & Validation...")
        
        error_cases = [
            # Invalid age
            {
                "name": "Invalid age (negative)",
                "data": {
                    "age": -5, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
                    "fbs": 0, "restecg": 0, "thalach": 120, "exang": 0,
                    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
                },
                "expected_status": 422
            },
            # Age > 150
            {
                "name": "Invalid age (too high)",
                "data": {
                    "age": 200, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
                    "fbs": 0, "restecg": 0, "thalach": 120, "exang": 0,
                    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
                },
                "expected_status": 422
            },
            # Invalid sex (should be 0-1)
            {
                "name": "Invalid sex",
                "data": {
                    "age": 50, "sex": 5, "cp": 0, "trestbps": 120, "chol": 200,
                    "fbs": 0, "restecg": 0, "thalach": 120, "exang": 0,
                    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
                },
                "expected_status": 422
            },
            # Missing required field
            {
                "name": "Missing required field",
                "data": {
                    "age": 50, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
                    "fbs": 0, "restecg": 0, "thalach": 120,
                    # missing exang, oldpeak, slope, ca, thal
                },
                "expected_status": 422
            },
            # Invalid cholesterol
            {
                "name": "Invalid cholesterol (too low)",
                "data": {
                    "age": 50, "sex": 1, "cp": 0, "trestbps": 120, "chol": 30,
                    "fbs": 0, "restecg": 0, "thalach": 120, "exang": 0,
                    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
                },
                "expected_status": 422
            }
        ]
        
        results = {"passed": 0, "failed": 0}
        
        for error_case in error_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json=error_case["data"],
                    timeout=5
                )
                
                if response.status_code == error_case["expected_status"]:
                    print(f"  ✓ {error_case['name']}: Got expected status {response.status_code}")
                    results["passed"] += 1
                else:
                    print(f"  ✗ {error_case['name']}: Expected {error_case['expected_status']}, got {response.status_code}")
                    results["failed"] += 1
            except Exception as e:
                print(f"  ✗ {error_case['name']}: Request failed: {str(e)}")
                results["failed"] += 1
        
        self.results["error_cases"].append(results)
        return results
    
    def test_malformed_input(self) -> Dict[str, int]:
        """Test handling of malformed requests"""
        print(f"\n[TEST] Malformed Input Handling...")
        
        test_cases = [
            {
                "name": "Empty JSON",
                "data": {},
                "endpoint": "/api/predict"
            },
            {
                "name": "String instead of JSON",
                "data": "not a valid json",
                "endpoint": "/api/predict",
                "raw": True
            },
            {
                "name": "Empty batch",
                "data": {"patients": []},
                "endpoint": "/api/predict/batch"
            },
            {
                "name": "Batch too large",
                "data": {"patients": [self.generate_patient(seed=i) for i in range(101)]},
                "endpoint": "/api/predict/batch"
            }
        ]
        
        results = {"validation_errors": 0, "other_errors": 0}
        
        for test_case in test_cases:
            try:
                if test_case.get("raw"):
                    response = requests.post(
                        f"{self.base_url}{test_case['endpoint']}",
                        data=test_case["data"],
                        timeout=5
                    )
                else:
                    response = requests.post(
                        f"{self.base_url}{test_case['endpoint']}",
                        json=test_case["data"],
                        timeout=5
                    )
                
                if response.status_code in [422, 400]:
                    print(f"  ✓ {test_case['name']}: Properly rejected with status {response.status_code}")
                    results["validation_errors"] += 1
                else:
                    print(f"  ⚠ {test_case['name']}: Got status {response.status_code}")
                    results["other_errors"] += 1
            except Exception as e:
                print(f"  ✓ {test_case['name']}: Request failed as expected: {type(e).__name__}")
                results["validation_errors"] += 1
        
        return results
    
    def test_extreme_values(self) -> Dict[str, int]:
        """Test handling of extreme but valid clinical values"""
        print(f"\n[TEST] Extreme Value Handling...")
        
        extreme_cases = [
            {
                "name": "Very young patient",
                "data": {
                    "age": 18, "sex": 0, "cp": 0, "trestbps": 90, "chol": 130,
                    "fbs": 0, "restecg": 0, "thalach": 200, "exang": 0,
                    "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 0
                }
            },
            {
                "name": "Very old patient",
                "data": {
                    "age": 120, "sex": 1, "cp": 3, "trestbps": 200, "chol": 560,
                    "fbs": 1, "restecg": 2, "thalach": 60, "exang": 1,
                    "oldpeak": 6.2, "slope": 2, "ca": 4, "thal": 3
                }
            },
            {
                "name": "High cholesterol",
                "data": {
                    "age": 50, "sex": 1, "cp": 0, "trestbps": 120, "chol": 560,
                    "fbs": 0, "restecg": 0, "thalach": 120, "exang": 0,
                    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
                }
            }
        ]
        
        results = {"success": 0, "failed": 0}
        
        for test_case in extreme_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json=test_case["data"],
                    timeout=5
                )
                
                if response.status_code == 200:
                    print(f"  ✓ {test_case['name']}: Prediction successful")
                    results["success"] += 1
                else:
                    print(f"  ⚠ {test_case['name']}: Status {response.status_code}")
                    results["failed"] += 1
            except Exception as e:
                print(f"  ✗ {test_case['name']}: {str(e)}")
                results["failed"] += 1
        
        return results
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        print(f"\n[TEST] Health Check Endpoint...")
        
        try:
            response = requests.get(
                f"{self.base_url}/api/health",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Health check passed")
                print(f"    - Status: {data.get('status')}")
                print(f"    - Model loaded: {data.get('model_loaded')}")
                print(f"    - Uptime: {data.get('uptime_seconds'):.2f}s")
                return True
            else:
                print(f"  ✗ Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"  ✗ Health check error: {str(e)}")
            return False
    
    def run_full_test_suite(self):
        """Run complete stress testing suite"""
        print("=" * 70)
        print("HEART DISEASE API - COMPREHENSIVE STRESS TEST SUITE")
        print("=" * 70)
        print(f"Target: {self.base_url}")
        print(f"Started: {datetime.utcnow().isoformat()}")
        
        # Check health first
        if not self.test_health_check():
            print("\n❌ API is not healthy. Aborting test suite.")
            return
        
        # Run tests
        self.test_single_prediction_latency(num_requests=100)
        self.test_batch_throughput(batch_sizes=[10, 25, 50, 100])
        self.test_error_handling()
        self.test_malformed_input()
        self.test_extreme_values()
        
        print("\n" + "=" * 70)
        print("TEST SUITE COMPLETED")
        print("=" * 70)
        
        # Save results
        self.save_results()
    
    def save_results(self, output_file: str = "stress_test_results.json"):
        """Save test results to file"""
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "api_url": self.base_url,
                "results": self.results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    tester = APIStressTester(base_url)
    tester.run_full_test_suite()
