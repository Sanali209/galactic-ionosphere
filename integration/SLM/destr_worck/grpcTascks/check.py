import socket


def check_tcp_connection(host, port):
    try:
        # Create a socket object
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Set a timeout for the connection attempt
            sock.settimeout(5)  # Timeout after 5 seconds
            # Attempt to connect to the host and port
            sock.connect((host, port))
            print(f"Successfully connected to {host}:{port}")
            return True
    except (socket.timeout, socket.error) as e:
        print(f"Failed to connect to {host}:{port} - {e}")
        return False

if __name__ == '__main__':
    host = '5.tcp.eu.ngrok.io'  # Extract the host from the address
    port = 19074  # Extract the port from the address
    check_tcp_connection(host, port)