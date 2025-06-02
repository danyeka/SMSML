import requests
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from datetime import datetime

class MLLoadTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = []
        self.errors = []
        
    def generate_sample_data(self):
        """Generate realistic sample data for credit scoring"""
        return {
            "RevolvingUtilizationOfUnsecuredLines": random.uniform(0, 1),
            "age": random.randint(18, 80),
            "NumberOfTime30-59DaysPastDueNotWorse": random.randint(0, 5),
            "DebtRatio": random.uniform(0, 2),
            "MonthlyIncome": random.uniform(1000, 15000),
            "NumberOfOpenCreditLinesAndLoans": random.randint(0, 20),
            "NumberOfTimes90DaysLate": random.randint(0, 3),
            "NumberRealEstateLoansOrLines": random.randint(0, 10),
            "NumberOfTime60-89DaysPastDueNotWorse": random.randint(0, 3),
            "NumberOfDependents": random.randint(0, 5)
        }
    
    def generate_batch_data(self, batch_size=5):
        """Generate batch of sample data"""
        return {
            "instances": [self.generate_sample_data() for _ in range(batch_size)]
        }
    
    def make_prediction_request(self, data, request_id=None):
        """Make a single prediction request"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            result = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "status_code": response.status_code,
                "latency": latency,
                "success": response.status_code == 200
            }
            
            if response.status_code == 200:
                result["response"] = response.json()
            else:
                result["error"] = response.text
                
            self.results.append(result)
            return result
            
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            
            error_result = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "status_code": 0,
                "latency": latency,
                "success": False,
                "error": str(e)
            }
            
            self.errors.append(error_result)
            return error_result
    
    def health_check(self):
        """Check if the service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def run_constant_load(self, duration_seconds=300, requests_per_second=2):
        """Run constant load test"""
        print(f"Starting constant load test: {requests_per_second} RPS for {duration_seconds} seconds")
        
        if not self.health_check():
            print("Service is not healthy. Aborting test.")
            return
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Single request
            if random.random() < 0.7:  # 70% single requests
                data = self.generate_sample_data()
            else:  # 30% batch requests
                data = self.generate_batch_data(random.randint(2, 5))
            
            self.make_prediction_request(data, f"const-{request_count}")
            request_count += 1
            
            # Sleep to maintain RPS
            time.sleep(1.0 / requests_per_second)
        
        print(f"Constant load test completed. Made {request_count} requests.")
    
    def run_spike_test(self, spike_duration=60, normal_rps=1, spike_rps=10):
        """Run spike test"""
        print(f"Starting spike test: {normal_rps} RPS normal, {spike_rps} RPS spike for {spike_duration}s")
        
        if not self.health_check():
            print("Service is not healthy. Aborting test.")
            return
        
        # Normal load for 2 minutes
        print("Phase 1: Normal load")
        self.run_constant_load(120, normal_rps)
        
        # Spike load
        print("Phase 2: Spike load")
        self.run_constant_load(spike_duration, spike_rps)
        
        # Back to normal
        print("Phase 3: Back to normal")
        self.run_constant_load(120, normal_rps)
        
        print("Spike test completed.")
    
    def run_concurrent_test(self, num_threads=5, requests_per_thread=20):
        """Run concurrent load test"""
        print(f"Starting concurrent test: {num_threads} threads, {requests_per_thread} requests each")
        
        if not self.health_check():
            print("Service is not healthy. Aborting test.")
            return
        
        def worker(thread_id):
            for i in range(requests_per_thread):
                data = self.generate_sample_data()
                self.make_prediction_request(data, f"thread-{thread_id}-req-{i}")
                time.sleep(random.uniform(0.1, 0.5))  # Random delay
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in futures:
                future.result()
        
        print("Concurrent test completed.")
    
    def run_error_injection_test(self, duration_seconds=180):
        """Run test with intentional errors to test error handling"""
        print(f"Starting error injection test for {duration_seconds} seconds")
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Generate different types of requests
            error_type = random.choice(["normal", "missing_field", "invalid_type", "empty_request"])
            
            if error_type == "normal":
                data = self.generate_sample_data()
            elif error_type == "missing_field":
                data = self.generate_sample_data()
                # Remove a random field
                field_to_remove = random.choice(list(data.keys()))
                del data[field_to_remove]
            elif error_type == "invalid_type":
                data = self.generate_sample_data()
                # Make a field invalid type
                field_to_corrupt = random.choice(list(data.keys()))
                data[field_to_corrupt] = "invalid_string"
            else:  # empty_request
                data = {}
            
            self.make_prediction_request(data, f"error-{request_count}")
            request_count += 1
            
            time.sleep(0.5)  # 2 RPS
        
        print(f"Error injection test completed. Made {request_count} requests.")
    
    def generate_report(self):
        """Generate test report"""
        if not self.results and not self.errors:
            print("No test results to report.")
            return
        
        all_results = self.results + self.errors
        df = pd.DataFrame(all_results)
        
        print("\n" + "="*50)
        print("LOAD TEST REPORT")
        print("="*50)
        
        print(f"Total Requests: {len(all_results)}")
        print(f"Successful Requests: {len(self.results)}")
        print(f"Failed Requests: {len(self.errors)}")
        print(f"Success Rate: {len(self.results)/len(all_results)*100:.2f}%")
        
        if len(self.results) > 0:
            successful_df = pd.DataFrame(self.results)
            latencies = successful_df['latency']
            
            print(f"\nLatency Statistics:")
            print(f"  Mean: {latencies.mean():.3f}s")
            print(f"  Median: {latencies.median():.3f}s")
            print(f"  95th Percentile: {latencies.quantile(0.95):.3f}s")
            print(f"  99th Percentile: {latencies.quantile(0.99):.3f}s")
            print(f"  Max: {latencies.max():.3f}s")
            print(f"  Min: {latencies.min():.3f}s")
        
        if len(self.errors) > 0:
            print(f"\nError Analysis:")
            error_df = pd.DataFrame(self.errors)
            if 'status_code' in error_df.columns:
                print("Status Code Distribution:")
                print(error_df['status_code'].value_counts())
        
        # Calculate throughput
        if len(all_results) > 1:
            timestamps = pd.to_datetime(df['timestamp'])
            duration = (timestamps.max() - timestamps.min()).total_seconds()
            throughput = len(all_results) / duration if duration > 0 else 0
            print(f"\nThroughput: {throughput:.2f} requests/second")
        
        print("\n" + "="*50)
    
    def run_comprehensive_test(self):
        """Run a comprehensive test suite"""
        print("Starting comprehensive load test suite...")
        
        # Test 1: Warm up
        print("\n1. Warm-up test (30 seconds, 1 RPS)")
        self.run_constant_load(30, 1)
        
        # Test 2: Normal load
        print("\n2. Normal load test (2 minutes, 2 RPS)")
        self.run_constant_load(120, 2)
        
        # Test 3: Concurrent test
        print("\n3. Concurrent test (3 threads, 10 requests each)")
        self.run_concurrent_test(3, 10)
        
        # Test 4: Spike test
        print("\n4. Spike test")
        self.run_spike_test(30, 1, 5)
        
        # Test 5: Error injection
        print("\n5. Error injection test (1 minute)")
        self.run_error_injection_test(60)
        
        # Generate final report
        self.generate_report()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Service Load Tester")
    parser.add_argument("--url", default="http://localhost:5000", help="Base URL of the ML service")
    parser.add_argument("--test-type", choices=["constant", "spike", "concurrent", "error", "comprehensive"], 
                       default="comprehensive", help="Type of test to run")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--rps", type=float, default=2.0, help="Requests per second")
    parser.add_argument("--threads", type=int, default=5, help="Number of concurrent threads")
    
    args = parser.parse_args()
    
    tester = MLLoadTester(args.url)
    
    if args.test_type == "constant":
        tester.run_constant_load(args.duration, args.rps)
    elif args.test_type == "spike":
        tester.run_spike_test(60, 1, args.rps)
    elif args.test_type == "concurrent":
        tester.run_concurrent_test(args.threads, 20)
    elif args.test_type == "error":
        tester.run_error_injection_test(args.duration)
    else:  # comprehensive
        tester.run_comprehensive_test()
    
    tester.generate_report()

if __name__ == "__main__":
    main()