#!/usr/bin/env python3
"""
WavePilot ChildProcess Worker

This module provides a JUCE ChildProcessWorker compatible interface for WavePilot.
It handles binary message communication with the host process via stdin/stdout.
"""

import sys
import struct
import json
import traceback
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from wavepilot_bridge import WavePilotBridge
    from logger import setup_logger
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}", file=sys.stderr)
    sys.exit(1)


class WavePilotWorker:
    """
    JUCE ChildProcessWorker compatible WavePilot bridge.

    Communicates with the host via binary messages over stdin/stdout following
    JUCE's MemoryBlock format:
    - Message length (4 bytes, little-endian)
    - Message data (JSON encoded as UTF-8)
    """

    def __init__(self):
        self.logger = setup_logger("WavePilot Worker")
        self.bridge = WavePilotBridge()
        self.initialized = False

    def send_message_to_coordinator(self, data):
        """Send a binary message to the coordinator process"""
        try:
            # Convert data to JSON bytes
            json_str = json.dumps(data)
            message_bytes = json_str.encode('utf-8')

            # Send length header (4 bytes, little-endian)
            length_header = struct.pack('<I', len(message_bytes))
            sys.stdout.buffer.write(length_header)

            # Send message data
            sys.stdout.buffer.write(message_bytes)
            sys.stdout.buffer.flush()

            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False

    def read_message_from_coordinator(self):
        """Read a binary message from the coordinator process"""
        try:
            # Read length header (4 bytes, little-endian)
            length_data = sys.stdin.buffer.read(4)
            if len(length_data) < 4:
                return None

            message_length = struct.unpack('<I', length_data)[0]

            # Read message data
            message_data = sys.stdin.buffer.read(message_length)
            if len(message_data) < message_length:
                return None

            # Decode JSON
            json_str = message_data.decode('utf-8')
            return json.loads(json_str)

        except Exception as e:
            self.logger.error(f"Failed to read message: {e}")
            return None

    def send_response(self, success, data=None, error=None, request_id=None):
        """Send a response message to the coordinator"""
        response = {
            "success": success,
            "data": data,
            "error": error
        }
        if request_id is not None:
            response["request_id"] = request_id

        return self.send_message_to_coordinator(response)

    def send_error(self, error_message, request_id=None):
        """Send error response"""
        return self.send_response(False, error=error_message, request_id=request_id)

    def send_success(self, data=None, request_id=None):
        """Send success response"""
        return self.send_response(True, data=data, request_id=request_id)

    def handle_message_from_coordinator(self, message):
        """Handle incoming message from coordinator"""
        try:
            command = message.get("command")
            params = message.get("params", {})
            request_id = message.get("request_id")

            self.logger.info(f"Received command: {command}")

            if command == "get_info":
                self.handle_get_info(params, request_id)
            elif command == "initialize":
                self.handle_initialize(params, request_id)
            elif command == "encode":
                self.handle_encode(params, request_id)
            elif command == "decode":
                self.handle_decode(params, request_id)
            elif command == "interpolate":
                self.handle_interpolate(params, request_id)
            elif command == "get_parameter_count":
                self.handle_get_parameter_count(params, request_id)
            elif command == "get_latent_bounds":
                self.handle_get_latent_bounds(params, request_id)
            else:
                self.send_error(f"Unknown command: {command}", request_id)

        except Exception as e:
            error_msg = f"Error handling command: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.send_error(error_msg, message.get("request_id"))

    def handle_get_info(self, params, request_id=None):
        """Handle WavePilot info request"""
        try:
            import sys
            import platform
            from pathlib import Path

            # Get version info
            info = {
                "name": "WavePilot Worker",
                "version": "2.0.0",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.system(),
                "arch": platform.machine(),
                "executable_path": str(Path(__file__).parent / "dist" / "wavepilot"),
                "communication": "JUCE ChildProcess binary messages",
                "features": [
                    "VAE parameter encoding/decoding",
                    "RBF interpolation",
                    "Latent space exploration",
                    "Hyperparameter optimization",
                    "Full optimization pipeline",
                    "Persistent worker process"
                ],
                "supported_commands": [
                    "initialize", "encode", "decode", "interpolate",
                    "get_parameter_count", "get_latent_bounds",
                    "load_optimization_data", "optimize_vae", "optimize_rbf",
                    "optimize_full_pipeline", "run_training_with_params", "get_info"
                ]
            }
            self.send_success(info, request_id)
        except Exception as e:
            self.send_error(f"Get info error: {str(e)}", request_id)

    def handle_initialize(self, params, request_id=None):
        """Handle initialization request"""
        try:
            model_path = params.get("model_path", "")
            dataset_path = params.get("dataset_path", "")

            self.logger.info(f"Initializing with model: {model_path}, dataset: {dataset_path}")

            success = self.bridge.initialize(model_path, dataset_path)
            if success:
                self.initialized = True
                self.send_success({"initialized": True}, request_id)
            else:
                self.send_error("Failed to initialize WavePilot", request_id)

        except Exception as e:
            self.send_error(f"Initialize error: {str(e)}", request_id)

    def handle_encode(self, params, request_id=None):
        """Handle parameter encoding request"""
        try:
            if not self.initialized:
                self.send_error("Not initialized", request_id)
                return

            parameters = params.get("parameters", [])
            latent_coords = self.bridge.encode_to_latent_space(parameters)

            if latent_coords:
                self.send_success({"latent_coordinates": latent_coords}, request_id)
            else:
                self.send_error("Encoding failed", request_id)

        except Exception as e:
            self.send_error(f"Encode error: {str(e)}", request_id)

    def handle_decode(self, params, request_id=None):
        """Handle latent space decoding request"""
        try:
            if not self.initialized:
                self.send_error("Not initialized", request_id)
                return

            latent_point = params.get("latent_point", [])
            parameters = self.bridge.decode_from_latent_space(latent_point)

            if parameters:
                self.send_success({"parameters": parameters}, request_id)
            else:
                self.send_error("Decoding failed", request_id)

        except Exception as e:
            self.send_error(f"Decode error: {str(e)}", request_id)

    def handle_interpolate(self, params, request_id=None):
        """Handle latent space interpolation request"""
        try:
            if not self.initialized:
                self.send_error("Not initialized", request_id)
                return

            start_latent = params.get("start_latent", [])
            end_latent = params.get("end_latent", [])
            t = params.get("t", 0.0)

            parameters = self.bridge.interpolate_latent_space(start_latent, end_latent, t)

            if parameters:
                self.send_success({"parameters": parameters}, request_id)
            else:
                self.send_error("Interpolation failed", request_id)

        except Exception as e:
            self.send_error(f"Interpolate error: {str(e)}", request_id)

    def handle_get_parameter_count(self, params, request_id=None):
        """Handle parameter count request"""
        try:
            count = self.bridge.get_parameter_count()
            self.send_success({"parameter_count": count}, request_id)
        except Exception as e:
            self.send_error(f"Get parameter count error: {str(e)}", request_id)

    def handle_get_latent_bounds(self, params, request_id=None):
        """Handle latent bounds request"""
        try:
            bounds = self.bridge.get_latent_bounds()
            self.send_success({"latent_bounds": bounds}, request_id)
        except Exception as e:
            self.send_error(f"Get latent bounds error: {str(e)}", request_id)

    def run(self):
        """Main worker loop - read messages from coordinator and respond"""
        self.logger.info("WavePilot Worker starting...")

        # Send initial connection message
        self.send_message_to_coordinator({
            "type": "worker_ready",
            "message": "WavePilot Worker initialized and ready"
        })

        try:
            while True:
                message = self.read_message_from_coordinator()
                if message is None:
                    # Connection lost or error
                    break

                self.handle_message_from_coordinator(message)

        except KeyboardInterrupt:
            self.logger.info("Worker interrupted")
        except Exception as e:
            self.logger.error(f"Worker error: {e}\n{traceback.format_exc()}")
        finally:
            self.logger.info("WavePilot Worker shutting down...")


def main():
    """Main entry point for the WavePilot worker process"""
    worker = WavePilotWorker()
    worker.run()


if __name__ == "__main__":
    main()