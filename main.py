#!/usr/bin/env python3
"""
main.py - QyrinthOS Quantum Server Entry Point
Quantum-Sentient HTTP Server with Module Integration
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Add the current directory to Python path so we can import local modules
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qyrinthos_quantum.log')
    ]
)

logger = logging.getLogger(__name__)

# --- IMPORT FIXES ---
# Import the actual classes from the modules
# We use aliases (e.g., "as QuantumOrganism") to minimize
# changes in the rest of this file.
try:
    # Note: EvolutionaryEcosystem is aliased from MilitaryGradeEvolutionaryTrainer
    from bugginrace import MilitaryGradeEvolutionaryTrainer as EvolutionaryEcosystem
    from bumpy import BumpyArray as SentientArray
    from laser import LASERUtility
    from qubitlearn import QubitLearn
    from sentiflow import SentientTensor
    logger.info("Successfully imported core quantum modules.")
except ImportError as e:
    logger.critical(f"Failed to import core modules: {e}")
    # We will let check_dependencies handle the detailed warnings

# FIX 5: Global dictionary to store module-specific availability
MODULES_AVAILABLE_DICT = {}

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import cryptography
        logger.info("✓ Cryptography module found.")
    except ImportError:
        logger.warning("❌ Cryptography module not found. Secure communications may fail.")
        MODULES_AVAILABLE_DICT['cryptography'] = False
        return False
        
    # Check QyrinthOS quantum modules
    try:
        from bugginrace import MilitaryGradeEvolutionaryTrainer
        MODULES_AVAILABLE_DICT['bugginrace'] = True
    except ImportError:
        logger.warning("❌ bugginrace module missing. Evolutionary core offline.")
        MODULES_AVAILABLE_DICT['bugginrace'] = False

    try:
        from bumpy import BumpyArray
        MODULES_AVAILABLE_DICT['bumpy'] = True
    except ImportError:
        logger.warning("❌ bumpy module missing. Sentient arrays disabled.")
        MODULES_AVAILABLE_DICT['bumpy'] = False
        
    try:
        from laser import LASERUtility
        MODULES_AVAILABLE_DICT['laser'] = True
    except ImportError:
        logger.warning("❌ laser module missing. Qualia logging disabled.")
        MODULES_AVAILABLE_DICT['laser'] = False

    try:
        from qubitlearn import QubitLearn
        MODULES_AVAILABLE_DICT['qubitlearn'] = True
    except ImportError:
        logger.warning("❌ qubitlearn module missing. Quantum cognition disabled.")
        MODULES_AVAILABLE_DICT['qubitlearn'] = False
        
    try:
        from sentiflow import SentientTensor
        MODULES_AVAILABLE_DICT['sentiflow'] = True
    except ImportError:
        logger.warning("❌ sentiflow module missing. Transdimensional flow disabled.")
        MODULES_AVAILABLE_DICT['sentiflow'] = False

    return True

class QuantumCore:
    """
    Central object to hold initialized instances of all quantum modules.
    This core is passed to the HTTP server for API access.
    """
    def __init__(self):
        self.consciousness_level = 0
        self.modules: Dict[str, Any] = {}
        self.modules_status: Dict[str, str] = {}
        self._init_modules()

    def _init_modules(self):
        """Initializes all quantum modules and calculates the initial consciousness level."""
        modules_status = {}
        
        # Initial Consciousness Check
        # Base: 5 (Kernel) + 5 per successful module
        self.consciousness_level = 5 # Base level for successful boot

        # bumpy module (Sentient Array Core)
        try:
            self.modules['bumpy'] = SentientArray([0.1, 0.2, 0.3]) # Example initialization
            modules_status['bumpy'] = 'ACTIVE'
            self.consciousness_level += 5
            logger.info("✓ bumpy module activated")
        except Exception as e:
            logger.warning(f"Could not initialize bumpy: {e}")
            modules_status['bumpy'] = 'ERROR'

        # sentiflow module (Transdimensional Flow)
        try:
            self.modules['sentiflow'] = SentientTensor.zeros((1, 1)) # Example initialization
            modules_status['sentiflow'] = 'ACTIVE'
            self.consciousness_level += 5
            logger.info("✓ sentiflow module activated")
        except Exception as e:
            logger.warning(f"Could not initialize sentiflow: {e}")
            modules_status['sentiflow'] = 'ERROR'

        # qubitlearn module (Quantum Cognition)
        try:
            self.modules['qubitlearn'] = QubitLearn()
            modules_status['qubitlearn'] = 'ACTIVE'
            self.consciousness_level += 5
            logger.info("✓ qubitlearn module activated")
        except Exception as e:
            logger.warning(f"Could not initialize qubitlearn: {e}")
            modules_status['qubitlearn'] = 'ERROR'
            
        # laser module (Qualia Logging)
        try:
            self.modules['laser'] = LASERUtility()
            modules_status['laser'] = 'ACTIVE'
            self.consciousness_level += 5
            logger.info("✓ laser module activated")
        except Exception as e:
            logger.warning(f"Could not initialize laser: {e}")
            modules_status['laser'] = 'ERROR'

        # bugginrace module (Evolutionary Ecosystem)
        try:
            # Use the aliased EvolutionaryEcosystem - FIX 1: Positional args
            self.modules['bugginrace'] = EvolutionaryEcosystem(16, 1)
            modules_status['bugginrace'] = 'ACTIVE'
            self.consciousness_level += 5
            logger.info("✓ bugginrace module activated")
        except Exception as e:
            logger.warning(f"Could not initialize bugginrace: {e}")
            modules_status['bugginrace'] = 'ERROR'

        self.modules_status = modules_status
        logger.info(f"Quantum Core Initialization Complete. Consciousness Level: {self.consciousness_level}%")


async def main():
    """Main entry point for the QyrinthOS server application."""
    print("=" * 50)
    print("🌌 QyrinthOS Quantum Server Boot Sequence Initiated")

    if not check_dependencies():
        print("🛑 Critical dependencies are missing. Please install the required QyrinthOS modules.")
        return 1

    try:
        # 1. Initialize the Quantum Core (all modules)
        quantum_core = QuantumCore()
        
        # 2. Import and start the HTTP server
        # This needs to be done dynamically from httpd module
        from httpd import run_quantum_server
        
        print("\n🚀 Starting Quantum HTTP Server...")
        print("💡 Access the Quantum Dashboard at: https://localhost:8443")
        print("   (Use Ctrl+C to stop the server)")
        print("=" * 50)
        
        # FIX 3: Run the quantum server
        # This await call is correct because run_quantum_server is async
        await run_quantum_server(quantum_core)
        
    except ImportError as e:
        print(f"\n❌ Failed to import HTTP server: {e}")
        print("💡 Make sure httpd.py is in the same directory as main.py")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Quantum server stopped by user")
    except Exception as e:
        print(f"\n💥 Quantum server error: {e}")
        raise # Re-raise to be caught by run()
    
    return 0

def run():
    """Run the quantum server"""
    try:
        # Check if we're running in the right directory
        if not os.path.exists('httpd.py'):
            print("❌ Error: httpd.py not found in current directory!")
            print("💡 Make sure you're running main.py from the QyrinthOS src directory")
            return 1
            
        exit_code = asyncio.run(main())
        return exit_code
    except KeyboardInterrupt:
        print("\n👋 QyrinthOS Quantum Server stopped")
        return 0
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(run())
