import json
import queue
import threading
from pythonosc import dispatcher, osc_server, udp_client
from logger import setup_logger

# Logger setup
logging = setup_logger('ReaLearn OSC Sender')

# Create a queue for incoming data packages
data_queue = queue.Queue()

# Global list to store OSC addresses
osc_addresses = []

def load_osc_addresses(file_path):
    """
    Load OSC addresses from a JSON file.
    Args:
        file_path (str): Path to the JSON file containing OSC addresses.
    Returns:
        list: A list of OSC addresses.
    """
    try:
        with open(file_path, 'r') as f:
            addresses = json.load(f)
        logging.info(f"Loaded {len(addresses)} OSC addresses from {file_path}")
        return addresses
    except Exception as e:
        logging.error(f"Error loading OSC addresses: {e}")
        return []

def process_data():
    """
    Process data from the queue and send OSC messages.
    """
    client = udp_client.SimpleUDPClient("localhost", 5106)  # OSC client to send messages
    while True:
        params = data_queue.get()
        if len(params) != len(osc_addresses):
            logging.error(f"Mismatch between params length ({len(params)}) and OSC addresses ({len(osc_addresses)})")
        else:
            for addr, value in zip(osc_addresses, params):
                client.send_message(addr, value)
                logging.info(f"Sent {value} to {addr}")

        data_queue.task_done()

def set_plugin_params(unused_addr, *args):
    """
    Receive parameters and add them to the processing queue.

    Args:
        unused_addr: The OSC address of the incoming message (not used).
        *args: The parameters received via OSC.
    """
    params = list(args)
    data_queue.put(params)
    logging.info(f"Received params: {params}")

def main():
    global osc_addresses

    # Load OSC addresses
    osc_addresses = load_osc_addresses("osc_addresses.json")
    if not osc_addresses:
        logging.error("No OSC addresses loaded. Exiting.")
        return

    # Set up OSC dispatcher and server
    dispatcher_map = dispatcher.Dispatcher()
    dispatcher_map.map('/interpolated_data', set_plugin_params)

    server = osc_server.ThreadingOSCUDPServer(('localhost', 5106), dispatcher_map)
    logging.info(f"Serving on: {server.server_address}")

    # Start the processing thread
    data_thread = threading.Thread(target=process_data, daemon=True)
    data_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down server.")
        server.shutdown()

if __name__ == "__main__":
    main()