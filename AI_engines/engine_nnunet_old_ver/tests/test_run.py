import time
import httpx
import sys

def test_endpoint():
    url = "http://127.0.0.1:8000/nnUNetTrainer_150epochs__nnUNetPlans__3d_fullres/run"
    
    # Wait for server to be up
    for _ in range(10):
        try:
            res = httpx.get("http://127.0.0.1:8000/health")
            if res.status_code == 200:
                print("Server is up!")
                break
        except Exception:
            time.sleep(1)
    else:
        print("Server failed to start")
        sys.exit(1)
        
    print("Testing endpoint:", url)
    try:
        with open("tests/mock_image.nii.gz", "rb") as f1, open("tests/mock_image.nii.gz", "rb") as f2:
            files = [
                ("files", ("mock_pet.nii.gz", f1, "application/octet-stream")),
                ("files", ("mock_ct.nii.gz", f2, "application/octet-stream"))
            ]
            response = httpx.post(url, files=files, timeout=60.0)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response size: {len(response.read())} bytes")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_endpoint()
