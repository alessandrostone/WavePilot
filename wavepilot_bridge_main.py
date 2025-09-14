#!/usr/bin/env python3
"""
WavePilot Bridge Executable Main

This is the main entry point for the standalone WavePilot executable.
It provides a simplified interface for C++ integration via subprocess communication.
"""

import sys
import json
import argparse
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


class WavePilotBridgeExecutable:
    """
    Standalone WavePilot bridge that communicates via JSON over stdin/stdout.
    This allows the C++ bridge to communicate with WavePilot without embedding Python.
    """
    
    def __init__(self):
        self.logger = setup_logger("WavePilot Bridge Executable")
        self.bridge = WavePilotBridge()
        self.initialized = False
        
    def send_response(self, success, data=None, error=None):
        """Send JSON response to stdout"""
        response = {
            "success": success,
            "data": data,
            "error": error
        }
        print(json.dumps(response), flush=True)
    
    def send_error(self, error_message):
        """Send error response"""
        self.send_response(False, error=error_message)
        
    def send_success(self, data=None):
        """Send success response"""
        self.send_response(True, data=data)
    
    def handle_initialize(self, params):
        """Handle initialization command"""
        try:
            dataset_path = params.get("dataset_path")
            model_path = params.get("model_path", "")
            
            if not dataset_path:
                self.send_error("dataset_path is required")
                return
                
            success = self.bridge.initialize(dataset_path, model_path if model_path else None)
            
            if success:
                self.initialized = True
                param_count = self.bridge.get_parameter_count()
                self.send_success({"parameter_count": param_count})
            else:
                self.send_error("Failed to initialize WavePilot bridge")
                
        except Exception as e:
            self.send_error(f"Initialize error: {str(e)}")
    
    def handle_encode(self, params):
        """Handle parameter encoding to latent space"""
        if not self.initialized:
            self.send_error("Bridge not initialized")
            return
            
        try:
            parameters = params.get("parameters", [])
            if not parameters:
                self.send_error("parameters list is required")
                return
                
            latent_coords = self.bridge.encode_to_latent(parameters)
            
            if latent_coords is not None:
                self.send_success({"latent_coords": latent_coords})
            else:
                self.send_error("Failed to encode parameters to latent space")
                
        except Exception as e:
            self.send_error(f"Encode error: {str(e)}")
    
    def handle_decode(self, params):
        """Handle latent space decoding to parameters"""
        if not self.initialized:
            self.send_error("Bridge not initialized")
            return
            
        try:
            latent_coords = params.get("latent_coords", [])
            if len(latent_coords) != 3:
                self.send_error("latent_coords must have exactly 3 elements")
                return
                
            parameters = self.bridge.decode_from_latent(latent_coords)
            
            if parameters is not None:
                self.send_success({"parameters": parameters})
            else:
                self.send_error("Failed to decode latent coordinates to parameters")
                
        except Exception as e:
            self.send_error(f"Decode error: {str(e)}")
    
    def handle_interpolate(self, params):
        """Handle latent space interpolation"""
        if not self.initialized:
            self.send_error("Bridge not initialized")
            return
            
        try:
            start_latent = params.get("start_latent", [])
            end_latent = params.get("end_latent", [])
            t = params.get("t", 0.0)
            
            if len(start_latent) != 3 or len(end_latent) != 3:
                self.send_error("start_latent and end_latent must have exactly 3 elements")
                return
                
            parameters = self.bridge.interpolate_latent(start_latent, end_latent, t)
            
            if parameters is not None:
                self.send_success({"parameters": parameters})
            else:
                self.send_error("Failed to interpolate in latent space")
                
        except Exception as e:
            self.send_error(f"Interpolate error: {str(e)}")
    
    def handle_get_parameter_count(self, params):
        """Handle parameter count request"""
        if not self.initialized:
            self.send_error("Bridge not initialized")
            return
            
        try:
            count = self.bridge.get_parameter_count()
            self.send_success({"parameter_count": count})
        except Exception as e:
            self.send_error(f"Get parameter count error: {str(e)}")
    
    def handle_get_latent_bounds(self, params):
        """Handle latent space bounds request"""
        if not self.initialized:
            self.send_error("Bridge not initialized")
            return

        try:
            bounds = self.bridge.get_latent_bounds()
            if bounds is not None:
                min_coords, max_coords = bounds
                self.send_success({
                    "min_coords": min_coords,
                    "max_coords": max_coords
                })
            else:
                self.send_error("Failed to get latent bounds")
        except Exception as e:
            self.send_error(f"Get latent bounds error: {str(e)}")

    def handle_load_optimization_data(self, params):
        """Handle loading data for optimization"""
        try:
            dataset_path = params.get("dataset_path")
            num_entries = params.get("num_entries")
            test_size = params.get("test_size", 0.2)
            mask_columns = params.get("mask_columns")

            if not dataset_path:
                self.send_error("dataset_path is required")
                return

            success = self.bridge.load_optimization_data(
                dataset_path, num_entries, test_size, mask_columns
            )

            if success:
                self.send_success({"message": "Optimization data loaded successfully"})
            else:
                self.send_error("Failed to load optimization data")

        except Exception as e:
            self.send_error(f"Load optimization data error: {str(e)}")

    def handle_optimize_vae(self, params):
        """Handle VAE hyperparameter optimization"""
        try:
            n_trials = params.get("n_trials", 100)

            best_params = self.bridge.optimize_vae(n_trials)

            if best_params is not None:
                self.send_success({"best_params": best_params})
            else:
                self.send_error("Failed to optimize VAE parameters")

        except Exception as e:
            self.send_error(f"VAE optimization error: {str(e)}")

    def handle_optimize_rbf(self, params):
        """Handle RBF hyperparameter optimization"""
        try:
            original_data = params.get("original_data")
            reduced_data = params.get("reduced_data")
            n_trials = params.get("n_trials", 100)

            if original_data is None or reduced_data is None:
                self.send_error("original_data and reduced_data are required")
                return

            best_params = self.bridge.optimize_rbf(original_data, reduced_data, n_trials)

            if best_params is not None:
                self.send_success({"best_params": best_params})
            else:
                self.send_error("Failed to optimize RBF parameters")

        except Exception as e:
            self.send_error(f"RBF optimization error: {str(e)}")

    def handle_optimize_full_pipeline(self, params):
        """Handle full optimization pipeline"""
        try:
            dataset_path = params.get("dataset_path")
            num_entries = params.get("num_entries")
            test_size = params.get("test_size", 0.2)
            mask_columns = params.get("mask_columns")
            vae_trials = params.get("vae_trials", 100)
            rbf_trials = params.get("rbf_trials", 100)

            if not dataset_path:
                self.send_error("dataset_path is required")
                return

            results = self.bridge.optimize_full_pipeline(
                dataset_path, num_entries, test_size, mask_columns, vae_trials, rbf_trials
            )

            if results is not None:
                self.send_success(results)
            else:
                self.send_error("Failed to run optimization pipeline")

        except Exception as e:
            self.send_error(f"Full pipeline optimization error: {str(e)}")

    def handle_run_training_with_params(self, params):
        """Handle training with specified parameters"""
        try:
            best_params = params.get("best_params")

            if not best_params:
                self.send_error("best_params is required")
                return

            vae_model = self.bridge.run_training_with_params(best_params)

            if vae_model is not None:
                # Convert model data to serializable format if needed
                self.send_success({"message": "Training completed successfully"})
            else:
                self.send_error("Failed to run training with parameters")

        except Exception as e:
            self.send_error(f"Training with parameters error: {str(e)}")

    def handle_get_info(self, params):
        """Handle WavePilot info request"""
        try:
            import sys
            import platform
            from pathlib import Path

            # Get version info
            info = {
                "name": "WavePilot Bridge",
                "version": "1.0.0",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.system(),
                "arch": platform.machine(),
                "executable_path": str(Path(__file__).parent / "dist" / "wavepilot"),
                "features": [
                    "VAE parameter encoding/decoding",
                    "RBF interpolation",
                    "Latent space exploration",
                    "Hyperparameter optimization",
                    "Full optimization pipeline"
                ],
                "supported_commands": [
                    "initialize", "encode", "decode", "interpolate",
                    "get_parameter_count", "get_latent_bounds",
                    "load_optimization_data", "optimize_vae", "optimize_rbf",
                    "optimize_full_pipeline", "run_training_with_params", "get_info"
                ]
            }

            self.send_success(info)

        except Exception as e:
            self.send_error(f"Get info error: {str(e)}")

    def process_command(self, command_json):
        """Process a single JSON command"""
        try:
            command = json.loads(command_json.strip())
            cmd_type = command.get("command")
            params = command.get("params", {})
            
            if cmd_type == "initialize":
                self.handle_initialize(params)
            elif cmd_type == "encode":
                self.handle_encode(params)
            elif cmd_type == "decode":
                self.handle_decode(params)
            elif cmd_type == "interpolate":
                self.handle_interpolate(params)
            elif cmd_type == "get_parameter_count":
                self.handle_get_parameter_count(params)
            elif cmd_type == "get_latent_bounds":
                self.handle_get_latent_bounds(params)
            elif cmd_type == "load_optimization_data":
                self.handle_load_optimization_data(params)
            elif cmd_type == "optimize_vae":
                self.handle_optimize_vae(params)
            elif cmd_type == "optimize_rbf":
                self.handle_optimize_rbf(params)
            elif cmd_type == "optimize_full_pipeline":
                self.handle_optimize_full_pipeline(params)
            elif cmd_type == "run_training_with_params":
                self.handle_run_training_with_params(params)
            elif cmd_type == "get_info":
                self.handle_get_info(params)
            elif cmd_type == "quit":
                self.send_success({"message": "Shutting down"})
                return False
            else:
                self.send_error(f"Unknown command: {cmd_type}")
                
        except json.JSONDecodeError as e:
            self.send_error(f"Invalid JSON: {str(e)}")
        except Exception as e:
            self.send_error(f"Command processing error: {str(e)}")
            
        return True
    
    def run_interactive_mode(self):
        """Run in interactive mode, reading commands from stdin"""
        self.logger.info("WavePilot Bridge Executable started in interactive mode")
        self.send_success({"message": "WavePilot Bridge Executable ready"})
        
        try:
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line:  # EOF
                        break
                        
                    if not self.process_command(line):
                        break
                        
                except EOFError:
                    break
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal")
                    break
                    
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            traceback.print_exc()
        finally:
            self.logger.info("WavePilot Bridge Executable shutting down")
    
    def run_single_command(self, command_json):
        """Run a single command and exit"""
        self.logger.info("WavePilot Bridge Executable started in single-command mode")
        self.process_command(command_json)


def main():
    """Main entry point"""
    # Check for JUCE ChildProcessWorker initialization first
    if len(sys.argv) > 1:
        # Try JUCE ChildProcessWorker initialization
        try:
            from wavepilot_worker import WavePilotWorker

            # Look for JUCE child process arguments or explicit worker flag
            command_line = " ".join(sys.argv[1:])

            # JUCE ChildProcessWorker uses specific argument patterns
            if any(arg.startswith(("--juce-child-process", "--child-process")) for arg in sys.argv) or "--worker" in sys.argv:
                print("INFO: Starting as JUCE ChildProcessWorker", file=sys.stderr)
                worker = WavePilotWorker()
                worker.run()
                return

        except ImportError:
            pass  # Fall back to original interface

    # Original command-line interface
    parser = argparse.ArgumentParser(description="WavePilot Bridge Executable")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode (read commands from stdin)")
    parser.add_argument("--command", "-c", type=str,
                       help="Execute a single JSON command and exit")
    parser.add_argument("--version", "-v", action="store_true",
                       help="Print version information")
    parser.add_argument("--worker", "-w", action="store_true",
                       help="Run as JUCE ChildProcessWorker (binary message mode)")

    args = parser.parse_args()

    if args.version:
        print("WavePilot Bridge Executable v2.0")
        return

    if args.worker:
        # JUCE ChildProcessWorker mode
        try:
            from wavepilot_worker import WavePilotWorker
            print("INFO: Starting as JUCE ChildProcessWorker", file=sys.stderr)
            worker = WavePilotWorker()
            worker.run()
        except ImportError as e:
            print(f"ERROR: Failed to import worker module: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Original JSON command interface
    bridge_exe = WavePilotBridgeExecutable()

    try:
        if args.command:
            # Single command mode
            bridge_exe.run_single_command(args.command)
        elif args.interactive or len(sys.argv) == 1:
            # Interactive mode (default)
            bridge_exe.run_interactive_mode()
        else:
            print("Error: No command specified. Use --interactive, --command, or --worker", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()