import requests
import json
import time
import statistics
import argparse
import io
import soundfile as sf
import numpy as np

def generate_test_audio():
    """Генерує тестове аудіо для benchmark"""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t) 
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer

def benchmark_api(api_url, num_requests=5):
    """Бенчмарк для API інференсу"""
    results = {
        'latencies': [],
        'success_count': 0,
        'error_count': 0,
        'errors': []
    }
    
    for i in range(num_requests):
        try:
            audio_buffer = generate_test_audio()
            
            files = {'file': ('test.wav', audio_buffer, 'audio/wav')}
            
            start_time = time.time()
            response = requests.post(f"{api_url}/predict", files=files, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                results['success_count'] += 1
                results['latencies'].append((end_time - start_time) * 1000)  # ms
                print(f"✅ Request {i+1}: {response.json().get('prediction', 'unknown')}")
            else:
                results['error_count'] += 1
                results['errors'].append(f"HTTP {response.status_code}")
                print(f"❌ Request {i+1}: HTTP {response.status_code}")
                
        except Exception as e:
            results['error_count'] += 1
            results['errors'].append(str(e))
            print(f"❌ Request {i+1}: {e}")
    
    if results['latencies']:
        results['avg_latency'] = statistics.mean(results['latencies'])
        results['min_latency'] = min(results['latencies'])
        results['max_latency'] = max(results['latencies'])
        if len(results['latencies']) >= 4:
            results['p95_latency'] = statistics.quantiles(results['latencies'], n=20)[18]
        else:
            results['p95_latency'] = results['avg_latency']
    
    results['success_rate'] = results['success_count'] / num_requests if num_requests > 0 else 0
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark ML API')
    parser.add_argument('--url', required=True, help='API URL')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--requests', type=int, default=5, help='Number of requests')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting benchmark for {args.url}")
    print(f"📊 Number of requests: {args.requests}")
    
    results = benchmark_api(args.url, args.requests)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Benchmark completed:")
    print(f"   Success rate: {results['success_rate']:.1%}")
    print(f"   Average latency: {results.get('avg_latency', 0):.2f}ms")
    print(f"   Results saved to: {args.output}")

if __name__ == '__main__':
    main()