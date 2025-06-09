import requests

def main():
    try:
        response = requests.get("https://subomi-api.onrender.com")
        print("Pinged successfully:", response.status_code)
    except Exception as e:
        print("Ping failed:", str(e))

if __name__ == "__main__":
    main()
