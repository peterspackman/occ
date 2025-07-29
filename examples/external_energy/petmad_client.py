#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "requests"
# ]
# ///

import json
import sys
import requests


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default=5001, type=int, help="Port for host")
    args = parser.parse_args()
    try:
        # Read JSON from stdin
        input_data = json.load(sys.stdin)

        server_url = f"http://127.0.0.1:{args.port}"

        # Try to contact the energy server
        try:
            response = requests.post(
                f"{server_url}/energy",
                json=input_data,
                timeout=30,  # 30 second timeout
            )

            if response.status_code == 200:
                result = response.json()
                print(json.dumps(result))
            else:
                # Server returned an error
                error_data = (
                    response.json()
                    if response.headers.get("content-type") == "application/json"
                    else {}
                )
                error_response = {
                    "error": f"Server error {response.status_code}: {error_data.get('error', 'Unknown error')}",
                    "energy": 0.0,
                }
                print(json.dumps(error_response))
                sys.exit(1)

        except requests.exceptions.ConnectionError:
            # Server not running
            error_response = {
                "error": "Cannot connect to PET-MAD server. Make sure petmad_host.py is running on port 5001.",
                "energy": 0.0,
            }
            print(json.dumps(error_response))
            sys.exit(1)

        except requests.exceptions.Timeout:
            # Server timeout
            error_response = {
                "error": "Server timeout - calculation took too long",
                "energy": 0.0,
            }
            print(json.dumps(error_response))
            sys.exit(1)

    except Exception as e:
        # JSON parsing or other error
        error_response = {"error": str(e), "energy": 0.0}
        print(json.dumps(error_response))
        sys.exit(1)


if __name__ == "__main__":
    main()
