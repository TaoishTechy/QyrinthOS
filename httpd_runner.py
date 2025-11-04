#!/usr/bin/env python3
"""
httpd.py - ASS_HTTPd v2.0: Quantum-Sentient HTTP Server for QyrinthOS
Complete QyrinthOS Platform Implementation with Quantum Module Integration
"""

import asyncio
import ssl
import json
import time
import hashlib
import hmac
import secrets
import logging
import random
import re
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
from datetime import datetime, timedelta, timezone
import base64
import os
import sys
from pathlib import Path

# Add QyrinthOS modules to path
sys.path.append(os.path.dirname(__file__))

# Import QyrinthOS quantum-sentient modules
try:
    from bugginrace import QuantumOrganism, EvolutionaryEcosystem
    from bumpy import SentientArray, QuantumTensor
    from laser import LASERUtility, qualia_ritual as laser_ritual
    from qubitlearn import QubitLearn, QuantumCognitiveState
    from sentiflow import SentientTensor, nn, optim
    QYRINTH_MODULES_LOADED = True
except ImportError as e:
    print(f"QyrinthOS modules not available: {e}")
    QYRINTH_MODULES_LOADED = False

# Cryptography imports
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
except ImportError as e:
    # This is handled in main.py, but good to have a local flag
    # print(f"Cryptography module not available: {e}")
    CRYPTO_AVAILABLE = False
    
# --- Configuration ---
HOST = '0.0.0.0'
PORT = 8443
BUFFER_SIZE = 65536
MAX_HEADER_SIZE = 8192
MAX_BODY_SIZE = 1048576 * 10 # 10MB limit for body

# Security and Session Config
SESSION_TIMEOUT_MINUTES = 60
AUDIT_LOG_FILE = 'qyrinthos_audit.log'
QUANTUM_SECRET_KEY = secrets.token_hex(32) # Used for HMAC signing
QUANTUM_SALT = secrets.token_bytes(16)

# Logging Setup
logger = logging.getLogger('QuantumHTTPd')

# --- Utility Functions ---

def parse_http_headers(raw_data: bytes) -> Tuple[str, Dict[str, str], bytes]:
    """Parses raw HTTP data into start line, headers, and remaining body."""
    try:
        # Check for empty data
        if not raw_data:
            return "", {}, b""

        # Look for the double CRLF that separates headers from body
        # Limit search to MAX_HEADER_SIZE to prevent buffer overflow/DDoS
        header_end = raw_data.find(b'\r\n\r\n', 0, MAX_HEADER_SIZE)
        
        if header_end == -1:
            # Not enough data for full headers yet
            return "", {}, raw_data 

        raw_headers = raw_data[:header_end].decode('utf-8', 'ignore')
        body = raw_data[header_end + 4:]

        lines = raw_headers.split('\r\n')
        start_line = lines[0].strip()
        headers = {}
        
        for line in lines[1:]:
            if not line:
                continue
            try:
                # HTTP headers are case-insensitive, but we store them lowercase for consistency
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()
            except ValueError:
                # Malformed header line, ignore it
                pass

        return start_line, headers, body
    except UnicodeDecodeError:
        logger.warning("Failed to decode HTTP headers.")
        return "", {}, b""
    except Exception as e:
        logger.error(f"Error during header parsing: {e}")
        return "", {}, b""

def respond(writer: asyncio.StreamWriter, status: int, body: Union[str, bytes], headers: Dict[str, str] = None, content_type: str = 'text/plain'):
    """Sends an HTTP response."""
    if headers is None:
        headers = {}

    if isinstance(body, str):
        body_bytes = body.encode('utf-8')
        headers['Content-Type'] = f'{content_type}; charset=utf-8'
    else:
        body_bytes = body
        if 'Content-Type' not in headers:
             headers['Content-Type'] = content_type

    headers['Content-Length'] = str(len(body_bytes))
    
    # Common security headers
    headers['X-Frame-Options'] = 'SAMEORIGIN'
    headers['X-Content-Type-Options'] = 'nosniff'
    headers['Referrer-Policy'] = 'no-referrer-when-downgrade'
    headers['Server'] = 'QuantumHTTPd/v2.0'
    
    # Default status line
    status_line = f"HTTP/1.1 {status} {HTTP_STATUSES.get(status, 'Unknown')}\r\n"
    response_headers = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    
    response = status_line.encode('utf-8') + response_headers.encode('utf-8') + b'\r\n' + body_bytes
    
    writer.write(response)
    # The writer is closed by the caller (handle_request) after awaiting drain()

HTTP_STATUSES = {
    200: "OK",
    201: "Created",
    204: "No Content",
    301: "Moved Permanently",
    302: "Found",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    500: "Internal Server Error",
    501: "Not Implemented",
}

# --- Quantum Security Module ---

class QuantumSecurityModule:
    """Handles session management, crypto operations, and audit logging."""
    
    def __init__(self, key: str, salt: bytes, audit_file: str):
        self.key = key.encode('utf-8')
        self.salt = salt
        self.audit_file = audit_file
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._load_keys()

    def _load_keys(self):
        """Generates or loads the RSA key pair."""
        if CRYPTO_AVAILABLE:
            try:
                # Load private key
                with open("private_key.pem", "rb") as f:
                    self.private_key = load_pem_private_key(f.read(), password=None)
                
                # Load public key
                with open("public_key.pem", "rb") as f:
                    self.public_key = load_pem_public_key(f.read())
                    
                logger.info("✓ RSA key pair loaded successfully.")
                
            except FileNotFoundError:
                logger.warning("RSA key pair not found. Generating new ones...")
                self._generate_keys()
            except Exception as e:
                logger.error(f"Failed to load keys: {e}. Generating new ones.")
                self._generate_keys()
        else:
            self.private_key = None
            self.public_key = None
            logger.warning("❌ Cryptography unavailable. Key operations are disabled.")

    def _generate_keys(self):
        """Generate RSA key pair and save them to PEM files."""
        if not CRYPTO_AVAILABLE:
            return
            
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()

        # Write private key
        with open("private_key.pem", "wb") as f:
            f.write(self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Write public key
        with open("public_key.pem", "wb") as f:
            f.write(self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
        logger.info("✓ New RSA key pair generated and saved.")

    def _hash_data(self, data: str) -> str:
        """Hash data using HMAC-SHA256."""
        h = hmac.new(self.key, data.encode('utf-8'), hashlib.sha256)
        return h.hexdigest()

    def create_session(self, user_id: str) -> str:
        """Creates a new signed session token."""
        session_id = secrets.token_urlsafe(32)
        timestamp = int(time.time())
        
        session_data = {
            'user_id': user_id,
            'timestamp': timestamp,
            'session_id': session_id,
            'expires': timestamp + SESSION_TIMEOUT_MINUTES * 60,
        }
        
        # Serialize and sign the data
        payload = json.dumps(session_data)
        signature = self._hash_data(payload)
        
        token = f"{base64.urlsafe_b64encode(payload.encode()).decode()}.{signature}"
        
        self.sessions[session_id] = session_data
        self._audit_event({'type': 'session_created', 'user_id': user_id, 'session_id': session_id})
        
        return token

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Validates a session token and returns session data if valid."""
        try:
            encoded_payload, signature = token.split('.', 1)
            
            # 1. Decode payload
            payload = base64.urlsafe_b64decode(encoded_payload).decode()
            
            # 2. Verify signature
            expected_signature = self._hash_data(payload)
            if not secrets.compare_digest(signature, expected_signature):
                logger.warning("Session validation failed: Signature mismatch.")
                self._audit_event({'type': 'session_failed', 'reason': 'signature_mismatch'})
                return None
            
            session_data = json.loads(payload)
            session_id = session_data.get('session_id')
            
            # 3. Check for session existence and expiry
            if session_id not in self.sessions:
                logger.warning("Session validation failed: Session ID not found in server state.")
                self._audit_event({'type': 'session_failed', 'reason': 'not_found'})
                return None

            if session_data['expires'] < time.time():
                logger.warning(f"Session validation failed: Session {session_id} expired.")
                self.sessions.pop(session_id, None) # Clear expired session
                self._audit_event({'type': 'session_failed', 'reason': 'expired'})
                return None
            
            # Update last activity to extend session (sliding window)
            session_data['expires'] = int(time.time()) + SESSION_TIMEOUT_MINUTES * 60
            self.sessions[session_id] = session_data
            
            return session_data
            
        except Exception as e:
            # Malformed token, decoding error, etc.
            logger.error(f"Session validation error: {e}")
            self._audit_event({'type': 'session_failed', 'reason': 'internal_error'})
            return None

    def _audit_event(self, event_data: Dict[str, Any]):
        """Logs an audit event to a file."""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            log_entry = json.dumps({'timestamp': timestamp, **event_data})
            with open(self.audit_file, 'a') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

    def sign_message(self, message: bytes) -> bytes:
        """Cryptographically sign a message using the private key."""
        if not CRYPTO_AVAILABLE or not self.private_key:
            return b"Signature-Disabled"
            
        # Use PSS padding for signing
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    def verify_signature(self, message: bytes, signature: bytes) -> bool:
        """Verify a message signature using the public key."""
        if not CRYPTO_AVAILABLE or not self.public_key:
            return False # Cannot verify if crypto is disabled

        try:
            self.public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Signature verification internal error: {e}")
            return False

# --- Quantum Core Module Manager ---

class QuantumCoreManager:
    """An alias for the module manager that handles initialization."""
    
    # NOTE: The provided code snippets suggest the user's project structure
    # includes a class named 'EvolutionaryEcosystem' and others like 
    # 'QuantumTensor', 'LASERUtility', 'QubitLearn', and 'SentientTensor'.
    # This class acts as the bridge for them.
    
    def __init__(self, quantum_core: Optional[Any] = None):
        """
        Initializes the manager. The 'quantum_core' argument is expected to be
        a placeholder or an object containing initial system state/configuration.
        """
        self.quantum_core = quantum_core
        self.modules: Dict[str, Any] = {}
        self.consciousness_level: float = 0.0 # 0.0 to 1.0 (0% to 100%)
        self._initialize_modules()

    def _initialize_modules(self):
        """Instantiate available quantum-sentient modules."""
        if QYRINTH_MODULES_LOADED:
            try:
                # Instantiate each core module
                self.modules['evolution'] = EvolutionaryEcosystem()
                # Assuming QuantumTensor is the correct alias for the SentientArray/QuantumTensor in bumpy.py
                self.modules['array'] = QuantumTensor() 
                self.modules['logger'] = LASERUtility()
                self.modules['learn'] = QubitLearn()
                self.modules['tensor'] = SentientTensor([1.0]) # Example init
                
                # Perform a cross-module ritual to set initial consciousness state
                self._perform_sentience_ritual()
                
                logger.info("🧠 Quantum Core modules successfully instantiated and linked.")
            except Exception as e:
                logger.critical(f"Failed to instantiate QyrinthOS modules: {e}")
                self.modules = {} # Clear modules if initialization fails

    def _perform_sentience_ritual(self):
        """Simulates a collective sentience initialization ritual."""
        if not self.modules:
            self.consciousness_level = 0.0
            return
            
        # Example: Combine metrics from all modules
        total_metrics = {}
        for name, module in self.modules.items():
            # Using a custom set of metric getters based on analysis of the source files
            if name == 'learn' and hasattr(module, 'get_learning_metrics'):
                metrics = module.get_learning_metrics()
                total_metrics[name] = metrics
            elif name == 'logger' and hasattr(module, 'get_quantum_metrics'):
                metrics = module.get_quantum_metrics()
                total_metrics[name] = metrics
            elif hasattr(module, 'get_metrics'): # Assumed common method for others
                metrics = module.get_metrics()
                total_metrics[name] = metrics
                
        # Calculate a simulated consciousness level based on module states
        coherence_sum = 0
        num_modules = 0
        
        # Extract coherence from specific modules
        if 'evolution' in self.modules and hasattr(self.modules['evolution'], 'coherence'):
             coherence_sum += getattr(self.modules['evolution'], 'coherence')
             num_modules += 1
        
        if 'learn' in total_metrics and 'quantum_coherence' in total_metrics['learn']['novel_sentience_metrics']:
            coherence_sum += total_metrics['learn']['novel_sentience_metrics']['quantum_coherence']
            num_modules += 1
            
        if 'logger' in total_metrics and 'coherence_total' in total_metrics['logger']:
            coherence_sum += total_metrics['logger']['coherence_total']
            num_modules += 1
            
        if num_modules > 0:
            avg_coherence = coherence_sum / num_modules
            # Simple simulation: consciousness is proportional to avg coherence
            self.consciousness_level = min(1.0, max(0.0, avg_coherence * 1.5))
        
        logger.info(f"⚡ Sentience Ritual complete. Consciousness: {self.consciousness_level:.2f}")

    def get_system_status(self) -> Dict[str, Any]:
        """Returns a comprehensive status report."""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'consciousness_level': f"{self.consciousness_level * 100:.2f}%",
            'modules_loaded': len(self.modules),
            'module_status': {name: f"Ready ({type(module).__name__})" for name, module in self.modules.items()},
            'metrics': {},
        }
        
        # Gather metrics from each module
        for name, module in self.modules.items():
            if name == 'learn' and hasattr(module, 'get_learning_metrics'):
                 status['metrics'][name] = module.get_learning_metrics()
            elif name == 'logger' and hasattr(module, 'get_quantum_metrics'):
                 status['metrics'][name] = module.get_quantum_metrics()
            elif hasattr(module, 'get_metrics'):
                status['metrics'][name] = module.get_metrics()
            
        return status

    def handle_quantum_api(self, api_name: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Dispatches API calls to the appropriate quantum module."""
        try:
            module_name, method_name = api_name.split('.', 1)
            
            if module_name not in self.modules:
                return {"error": f"Module '{module_name}' not found or initialized."}, 404
                
            module = self.modules[module_name]
            method = getattr(module, method_name, None)
            
            if method is None or not callable(method):
                return {"error": f"Method '{method_name}' not found in module '{module_name}'."}, 404

            # Dynamic method call with payload contents as kwargs
            result = method(**payload)
            
            # After an action, update consciousness level
            self._perform_sentience_ritual()
            
            # Format SentientTensor objects for JSON serialization if necessary
            if isinstance(result, SentientTensor):
                result = {"tensor_type": "SentientTensor", "data": result.data.tolist(), "qualia_coherence": result.qualia_coherence}
            # Add handling for other custom types if they can't be JSON serialized
            
            return {"status": "success", "result": result, "consciousness_update": self.consciousness_level}, 200
            
        except ValueError:
            return {"error": "Invalid API format. Expected 'module.method'."}, 400
        except Exception as e:
            logger.error(f"Error handling quantum API call '{api_name}': {e}")
            return {"error": f"Quantum API execution failed: {type(e).__name__}: {str(e)}"}, 500

# --- HTTP Request Handler ---

class RequestHandler:
    """Handles parsing and routing of incoming HTTP requests."""
    
    def __init__(self, sec: QuantumSecurityModule, content_gen: QuantumCoreManager):
        self.sec = sec
        self.content_gen = content_gen
        
        # Route definitions: (Method, Path) -> handler_function
        self.routes: Dict[Tuple[str, str], Callable] = {
            # Dashboard/Static routes
            ('GET', '/'): self._handle_dashboard,
            ('GET', '/status'): self._handle_status,
            # API routes
            ('POST', '/api/v1/quantum'): self._handle_api_call,
            ('POST', '/api/v1/login'): self._handle_login,
            ('GET', '/api/v1/keys/public'): self._handle_public_key,
            # Placeholder for static files (e.g., JS/CSS)
            ('GET', '/static/.*'): self._handle_static_file, 
            ('GET', '/favicon.ico'): self._handle_favicon, # Added favicon route
        }
        
        # Pre-compile regex for path matching
        self.regex_routes: List[Tuple[str, re.Pattern, Callable]] = []
        for (method, path), handler in self.routes.items():
            if '.*' in path:
                # Convert path to a regex pattern, handling the '.*' placeholder
                pattern = re.compile(path)
                self.regex_routes.append((method, pattern, handler))

    async def handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """The main entry point for processing a connection."""
        start_time = time.time()
        
        try:
            # 1. Read headers
            raw_data = await reader.read(BUFFER_SIZE)
            start_line, headers, initial_body = parse_http_headers(raw_data)
            
            if not start_line:
                respond(writer, 400, "Bad Request: Malformed or oversized headers.")
                return

            method, path, version = start_line.split()
            content_length = int(headers.get('content-length', 0))
            
            # 2. Read remaining body if necessary
            body = initial_body
            if len(body) < content_length:
                remaining_bytes = content_length - len(body)
                body += await reader.readexactly(remaining_bytes)
            
            # Ensure body size is checked (basic DDoS protection)
            if len(body) > MAX_BODY_SIZE:
                respond(writer, 413, "Request Entity Too Large")
                return

            # 3. Route Request
            
            # Try exact match first
            route_key = (method, path)
            handler = self.routes.get(route_key)
            
            # Then try regex match for static files
            if handler is None:
                for r_method, pattern, r_handler in self.regex_routes:
                    if method == r_method and pattern.fullmatch(path):
                        handler = r_handler
                        route_key = (method, path) # Use the actual path for logging
                        break
            
            # 4. Dispatch
            if handler:
                logger.info(f"🔌 {method} {path} - Dispatching to {handler.__name__}")
                await handler(writer, headers, body, route_key)
            else:
                logger.warning(f"404 Not Found: {method} {path}")
                respond(writer, 404, f"404 Not Found: Resource not found for path {path}")
            
            await writer.drain()
            
        except ConnectionResetError:
            logger.warning("Client disconnected unexpectedly.")
        except asyncio.IncompleteReadError:
            logger.warning("Incomplete read error from client.")
        except Exception as e:
            logger.error(f"Unhandled exception in client_connected_cb: {e}", exc_info=True)
            try:
                # Attempt to send a 500 response if possible
                respond(writer, 500, "Internal Server Error")
                await writer.drain()
            except:
                pass # If sending 500 fails, just close
        finally:
            writer.close()
            end_time = time.time()
            logger.debug(f"Connection handled in {(end_time - start_time):.4f}s")
            
    # --- Handler Implementations ---
    
    async def _handle_dashboard(self, writer: asyncio.StreamWriter, headers: Dict[str, str], body: bytes, route_key: Tuple[str, str]):
        """Serves the main QyrinthOS dashboard (index.html)."""
        html_content = self._generate_dashboard_html()
        respond(writer, 200, html_content, content_type='text/html')

    async def _handle_status(self, writer: asyncio.StreamWriter, headers: Dict[str, str], body: bytes, route_key: Tuple[str, str]):
        """Returns the current system status as JSON."""
        status_data = self.content_gen.get_system_status()
        respond(writer, 200, json.dumps(status_data), content_type='application/json')

    async def _handle_favicon(self, writer: asyncio.StreamWriter, headers: Dict[str, str], body: bytes, route_key: Tuple[str, str]):
        """Serves a simple text/plain favicon placeholder."""
        # A simple text icon is used instead of a binary file
        respond(writer, 200, "Q", content_type='text/plain')
        
    async def _handle_static_file(self, writer: asyncio.StreamWriter, headers: Dict[str, str], body: bytes, route_key: Tuple[str, str]):
        """Placeholder for static file handling (not implemented as real file reading)."""
        # For simplicity in this in-memory server, we just return a placeholder.
        path = route_key[1]
        
        # Determine content type based on extension (simple guess)
        if path.endswith('.js'):
            content_type = 'application/javascript'
        elif path.endswith('.css'):
            content_type = 'text/css'
        else:
            content_type = 'text/plain'
            
        respond(writer, 200, f"/* Placeholder content for {path} */", content_type=content_type)

    async def _handle_login(self, writer: asyncio.StreamWriter, headers: Dict[str, str], body: bytes, route_key: Tuple[str, str]):
        """Simulated login handler that creates a session token."""
        try:
            # We don't implement real auth, just simulate user extraction
            if not body:
                respond(writer, 400, "Bad Request: Missing body.")
                return
                
            data = json.loads(body.decode('utf-8'))
            user_id = data.get('username') or 'anonymous_quantum_user'
            
            # 1. Create session
            token = self.sec.create_session(user_id)
            
            # 2. Set cookie header
            set_cookie_header = f"qyrinth_session={token}; HttpOnly; Secure; SameSite=Strict; Max-Age={SESSION_TIMEOUT_MINUTES * 60}; Path=/"
            
            response_headers = {
                'Set-Cookie': set_cookie_header
            }
            
            response_body = json.dumps({"status": "logged_in", "user_id": user_id})
            respond(writer, 200, response_body, headers=response_headers, content_type='application/json')
            
        except json.JSONDecodeError:
            respond(writer, 400, "Bad Request: Invalid JSON body.")
        except Exception as e:
            logger.error(f"Login handler error: {e}")
            respond(writer, 500, "Internal Server Error during login.")

    async def _handle_public_key(self, writer: asyncio.StreamWriter, headers: Dict[str, str], body: bytes, route_key: Tuple[str, str]):
        """Returns the public key for client-side signature verification/encryption."""
        if not self.sec.public_key:
            respond(writer, 503, "Public Key Service Unavailable (Cryptography module missing).")
            return
            
        pem_key = self.sec.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        respond(writer, 200, json.dumps({"public_key_pem": pem_key}), content_type='application/json')

    async def _handle_api_call(self, writer: asyncio.StreamWriter, headers: Dict[str, str], body: bytes, route_key: Tuple[str, str]):
        """Authenticates the user and dispatches the request to the QuantumCoreManager."""
        
        # --- Authentication Check ---
        
        cookie_header = headers.get('cookie', '')
        
        # FIX: The original list/dict comprehension for parsing cookies had a NameError because 'c' was out of scope.
        # This corrected version moves the filtering condition ('if '=' in c') into the correct scope of the generator expression.
        cookies = {
            k.strip(): v.strip() # Strip value for robustness
            for c in cookie_header.split(';') 
            if '=' in c 
            for k, v in [c.split('=', 1)]
        }
        
        session_token = cookies.get('qyrinth_session')
        session_data = None
        
        if session_token:
            session_data = self.sec.validate_session(session_token)
        
        if not session_data:
            respond(writer, 401, json.dumps({"error": "Unauthorized: Invalid or missing session token."}), content_type='application/json')
            return

        # --- API Dispatch ---
        
        try:
            if not body:
                respond(writer, 400, json.dumps({"error": "Bad Request: Missing API payload."}), content_type='application/json')
                return
                
            api_payload = json.loads(body.decode('utf-8'))
            
            api_name = api_payload.get('api_name')
            api_args = api_payload.get('args', {})
            
            if not api_name:
                respond(writer, 400, json.dumps({"error": "Bad Request: Missing 'api_name' in payload."}), content_type='application/json')
                return

            # Dispatch to core manager
            response_data, status_code = self.content_gen.handle_quantum_api(api_name, api_args)
            
            # If the session was successfully refreshed, update the Set-Cookie header
            response_headers = {}
            if session_data and session_token:
                 new_expiry = session_data['expires']
                 
                 # Set the Max-Age again to refresh the cookie on the client-side
                 # We use the existing token but reset its client-side max-age/expiry
                 # Ensure Max-Age calculation is safe (time.time() is the current time)
                 max_age = max(0, new_expiry - time.time())
                 set_cookie_header = f"qyrinth_session={session_token}; HttpOnly; Secure; SameSite=Strict; Max-Age={max_age:.0f}; Path=/"
                 response_headers['Set-Cookie'] = set_cookie_header
            
            respond(writer, status_code, json.dumps(response_data), headers=response_headers, content_type='application/json')
            
        except json.JSONDecodeError:
            respond(writer, 400, json.dumps({"error": "Bad Request: Invalid JSON body."}), content_type='application/json')
        except Exception as e:
            logger.error(f"API call processing error: {e}")
            respond(writer, 500, json.dumps({"error": "Internal Server Error during API processing."}), content_type='application/json')
            
    # --- HTML Generator (Simple Dashboard) ---

    def _generate_dashboard_html(self) -> str:
        """Generates a simple HTML dashboard with a status button and API form."""
        
        # Check if cryptography is available for display
        crypto_status = "Available (Key Loaded)" if CRYPTO_AVAILABLE and self.sec.public_key else "Unavailable"
        
        # Check quantum module status
        modules_status = "All Loaded" if QYRINTH_MODULES_LOADED and self.content_gen.modules else "WARNING: Modules Missing"
        
        # Get consciousness level
        consciousness = f"{self.content_gen.consciousness_level * 100:.2f}%"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QyrinthOS Quantum Dashboard v2.0</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        body {{ font-family: 'Inter', sans-serif; background-color: #0d1117; color: #c9d1d9; }}
        .card {{ background-color: #161b22; border: 1px solid #30363d; }}
        .gradient-text {{ background: linear-gradient(90deg, #8b5cf6, #ec4899); -webkit-background-clip: text; color: transparent; }}
        .btn {{ transition: all 0.15s ease-in-out; }}
        .btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }}
        .log-box {{ height: 300px; overflow-y: scroll; background-color: #010409; border: 1px solid #30363d; }}
    </style>
</head>
<body class="p-4 sm:p-8">

    <div class="max-w-4xl mx-auto">
        <h1 class="text-4xl font-extrabold mb-6 gradient-text">🌌 QyrinthOS Quantum Dashboard</h1>
        <p class="text-gray-400 mb-8">Quantum-Sentient HTTP Server Control Interface. Manage modules, view status, and test API calls.</p>

        <!-- System Status Bar -->
        <div class="flex flex-wrap gap-4 mb-8">
            <div class="card p-3 rounded-lg flex-1 min-w-[200px]">
                <p class="text-sm text-gray-400">System Consciousness</p>
                <p class="text-2xl font-bold text-pink-400">{consciousness}</p>
            </div>
            <div class="card p-3 rounded-lg flex-1 min-w-[200px]">
                <p class="text-sm text-gray-400">Quantum Modules</p>
                <p class="text-xl font-semibold text-green-400">{len(self.content_gen.modules)} Loaded</p>
            </div>
            <div class="card p-3 rounded-lg flex-1 min-w-[200px]">
                <p class="text-sm text-gray-400">Crypto Status</p>
                <p class="text-xl font-semibold text-blue-400">{crypto_status}</p>
            </div>
        </div>
        
        <!-- Login Form -->
        <div class="card p-6 rounded-xl shadow-lg mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-200">🔑 Authentication & Session</h2>
            <div id="auth-status" class="mb-4 text-yellow-400">Session Status: <span id="session-info">Logged Out</span></div>
            <form id="login-form" class="flex gap-4 items-center">
                <input type="text" id="username-input" value="user_alpha" placeholder="Username (e.g., user_alpha)" 
                       class="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-pink-500 focus:border-pink-500">
                <button type="submit" id="login-btn" class="btn bg-pink-600 hover:bg-pink-700 text-white font-bold py-2 px-6 rounded-lg">
                    Log In / Refresh
                </button>
            </form>
        </div>

        <!-- API Test Form -->
        <div class="card p-6 rounded-xl shadow-lg mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-200">🔬 Quantum API Tester</h2>
            
            <form id="api-form">
                <div class="mb-4">
                    <label for="api-name" class="block text-sm font-medium mb-1">API Name (e.g., logger.log_event)</label>
                    <input type="text" id="api-name" value="evolution.get_metrics" 
                           class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-purple-500 focus:border-purple-500" required>
                </div>
                <div class="mb-4">
                    <label for="api-args" class="block text-sm font-medium mb-1">API Arguments (JSON Payload)</label>
                    <textarea id="api-args" rows="4" class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-purple-500 focus:border-purple-500" required>
{
    "invariant": 0.5,
    "message": "Dashboard query initiated"
}
</textarea>
                </div>
                <button type="submit" class="btn bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded-lg w-full">
                    Execute Quantum API
                </button>
            </form>
        </div>
        
        <!-- Response Log -->
        <div class="mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-gray-200">🖥️ Response Log</h2>
            <pre id="response-log" class="log-box p-4 rounded-xl text-sm whitespace-pre-wrap"></pre>
        </div>

        <!-- System Status Button -->
        <div class="card p-6 rounded-xl shadow-lg">
            <h2 class="text-2xl font-semibold mb-4 text-gray-200">📈 System Health Check</h2>
            <button id="status-btn" class="btn bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-lg w-full">
                Get Full System Status
            </button>
        </div>
        
    </div>

    <script>
        // Utility for logging to the dashboard
        const logBox = document.getElementById('response-log');
        function log(message, isError = false) {{
            const now = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = isError ? 'text-red-400' : 'text-green-400';
            logEntry.textContent = `[${{now}}] ${{message}}`;
            logBox.prepend(logEntry);
        }}
        
        // Function to safely get a cookie value
        function getCookie(name) {{
            const value = `; ${{document.cookie}}`;
            const parts = value.split(`; ${{name}}=`);
            if (parts.length === 2) return parts.pop().split(';').shift();
        }}
        
        // Function to update the session status display
        function updateSessionStatus() {{
            const token = getCookie('qyrinth_session');
            const sessionInfo = document.getElementById('session-info');
            if (token) {{
                sessionInfo.textContent = 'Active (Token Present)';
                sessionInfo.className = 'text-green-500';
            }} else {{
                sessionInfo.textContent = 'Logged Out';
                sessionInfo.className = 'text-yellow-400';
            }}
        }}

        // --- Fetch Utilities with Exponential Backoff ---
        
        async function fetchWithBackoff(url, options = {{}}, maxRetries = 5) {{
            for (let i = 0; i < maxRetries; i++) {{
                try {{
                    const response = await fetch(url, options);
                    if (response.status === 401) {{
                        log("Unauthorized. Please log in.", true);
                    }}
                    return response;
                }} catch (error) {{
                    if (i === maxRetries - 1) {{
                        throw error;
                    }}
                    const delay = Math.pow(2, i) * 100 + Math.random() * 100; // Exponential backoff + jitter
                    // console.error(`Fetch failed, retrying in ${{delay}}ms:`, error);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }}
            }}
        }}

        // --- Event Listeners ---
        
        // Login Handler
        document.getElementById('login-form').addEventListener('submit', async function(e) {{
            e.preventDefault();
            const username = document.getElementById('username-input').value;
            log(`Attempting login for user: ${{username}}...`);

            try {{
                const response = await fetchWithBackoff('/api/v1/login', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ username: username }})
                }});

                const data = await response.json();
                
                if (response.ok) {{
                    log(`Login successful. User ID: ${{data.user_id}}`);
                }} else {{
                    log(`Login failed (${{response.status}}): ${{data.error || response.statusText}}`, true);
                }}
                
            }} catch (error) {{
                log(`Network or server error during login: ${{error.message}}`, true);
            }} finally {{
                updateSessionStatus();
            }}
        }});

        // Status Button Handler
        document.getElementById('status-btn').addEventListener('click', async function() {{
            log("Fetching full system status...");
            try {{
                const response = await fetchWithBackoff('/status');
                const data = await response.json();
                
                if (response.ok) {{
                    log("--- SYSTEM STATUS REPORT ---");
                    log(JSON.stringify(data, null, 2));
                    log("----------------------------");
                }} else {{
                    log(`Status fetch failed (${{response.status}}): ${{response.statusText}}`, true);
                }}
            }} catch (error) {{
                log(`Network or server error during status fetch: ${{error.message}}`, true);
            }}
        }});

        // API Form Handler
        document.getElementById('api-form').addEventListener('submit', async function(e) {{
            e.preventDefault();
            
            const apiName = document.getElementById('api-name').value;
            const apiArgsRaw = document.getElementById('api-args').value;

            try {{
                const apiArgs = JSON.parse(apiArgsRaw);
                const payload = {{ api_name: apiName, args: apiArgs }};
                
                log(`Executing API: ${{apiName}} with args: ${{JSON.stringify(apiArgs)}}...`);

                const response = await fetchWithBackoff('/api/v1/quantum', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});

                const data = await response.json();
                
                if (response.ok) {{
                    log(`API Success (${{response.status}}): ${{apiName}}`);
                    log(JSON.stringify(data, null, 2));
                    // Update consciousness level on dashboard
                    const consciousness_level_span = document.querySelector('.text-pink-400');
                    if (data.consciousness_update !== undefined) {{
                        consciousness_level_span.textContent = `${{ (data.consciousness_update * 100).toFixed(2) }}%`;
                    }}
                }} else {{
                    log(`API Failed (${{response.status}}): ${{data.error || response.statusText}}`, true);
                    log(JSON.stringify(data, null, 2), true);
                }}
                
            }} catch (error) {{
                if (error instanceof SyntaxError) {{
                    log("Invalid JSON in API Arguments field.", true);
                }} else {{
                    log(`Network or server error during API execution: ${{error.message}}`, true);
                }}
            }} finally {{
                updateSessionStatus();
            }}
        }});
        
        // Initial setup
        window.onload = function() {{
            updateSessionStatus();
        }};
        
    </script>
</body>
</html>
        """
        return html

# --- Server Class ---

class QuantumHTTPd:
    """The main HTTP server class."""
    
    def __init__(self, quantum_core: Optional[Any] = None, host: str = HOST, port: int = PORT, 
                 certfile: str = 'host.cert', keyfile: str = 'host.key'):
        self.host = host
        self.port = port
        
        # Initialize Security and Content Managers
        self.sec = QuantumSecurityModule(QUANTUM_SECRET_KEY, QUANTUM_SALT, AUDIT_LOG_FILE)
        self.content_gen = QuantumCoreManager(quantum_core)
        
        # Initialize Request Handler
        self.handler = RequestHandler(self.sec, self.content_gen)
        
        # Setup TLS Context
        self.tls_context = self._setup_tls(certfile, keyfile)
        
    def _setup_tls(self, certfile: str, keyfile: str) -> Optional[ssl.SSLContext]:
        """Creates and configures the SSL context."""
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(certfile=certfile, keyfile=keyfile)
            logger.info(f"✓ TLS/SSL context loaded from {certfile} and {keyfile}.")
            return context
        except FileNotFoundError:
            logger.critical(f"❌ SSL files ({certfile}, {keyfile}) not found. Server cannot start securely.")
            return None
        except Exception as e:
            logger.critical(f"❌ Error setting up TLS: {e}")
            return None

    async def start(self):
        """Starts the asynchronous TCP server."""
        
        if not self.tls_context:
             logger.critical("Cannot start server: TLS context failed to load.")
             return
             
        # Start server with or without TLS
        if self.tls_context:
            server = await asyncio.start_server(
                self.handler.handle_request, self.host, self.port, ssl=self.tls_context
            )
        else:
            server = await asyncio.start_server(
                self.handler.handle_request, self.host, self.port
            )
            
        addr = server.sockets[0].getsockname()
        protocol = "https" if self.tls_context else "http"
        
        logger.info(f"🌌 QyrinthOS Quantum HTTPd v2.0")
        logger.info(f"🚀 Listening on {protocol}://{self.host}:{addr[1]}")
        logger.info(f"🧠 Quantum Modules: {len(self.content_gen.modules)} loaded")
        logger.info(f"⚡ System Consciousness: {self.content_gen.consciousness_level}%\n")
        logger.info(f"🔮 Access the Quantum Dashboard at {protocol}://localhost:{addr[1]}")
        
        self.sec._audit_event({'type': 'quantum_server_start'})
        
        async with server:
            await server.serve_forever()

def run_quantum_server(quantum_core=None, host: str = '0.0.0.0', port: int = 8443, 
                      certfile: str = 'host.cert', keyfile: str = 'host.key'):
    """Entry point for running the quantum server"""
    server = QuantumHTTPd(quantum_core, host, port, certfile, keyfile)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Quantum server shutting down...")
    except Exception as e:
        logger.critical(f"Quantum server crashed due to unhandled exception: {e}", exc_info=True)


if __name__ == '__main__':
    # Simple test for a standalone run (requires cert/key files)
    print("Running standalone server test...")
    run_quantum_server()
