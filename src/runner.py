import json
import time

class LLMEvaluatorHarness:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate(self, model_endpoint):
        data = self.load_data()
        results = []
        
        for item in data:
            start_time = time.time()
            response = f"Mock response for: {item['prompt']}"
            latency = time.time() - start_time
            
            results.append({
                "id": item["id"],
                "latency_sec": round(latency, 4),
                "success": True,
                "response": response
            })
        return results

if __name__ == "__main__":
    print("LLM Evaluator Harness Initialized.")
