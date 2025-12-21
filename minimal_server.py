import socket
import sys
import time

PORT = 50008 # Use a different port to avoid conflicts
READY_MESSAGE = "Minimal socket server ready on port"

def run_minimal_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", PORT))
        except OSError as e:
            print(f"ERROR: Could not bind to port {PORT}: {e}", file=sys.stderr)
            sys.exit(1)
            
        sock.listen(1)
        print(f"{READY_MESSAGE} {PORT}", flush=True, file=sys.stderr)
        
        while True:
            conn, addr = sock.accept()
            with conn:
                data = conn.recv(1024).decode("utf-8").strip()
                response = f"Echo: {data}\n"
                conn.sendall(response.encode("utf-8"))

if __name__ == "__main__":
    run_minimal_server()
