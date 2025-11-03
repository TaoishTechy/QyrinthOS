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

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import cryptography
        logger.info("✓ Cryptography library available")
    except ImportError:
        logger.warning("⚠ Cryptography library not available - using simplified security")
    
    try:
        # Try to import QyrinthOS modules
        from bugginrace import EvolutionaryEcosystem
        from bumpy import SentientArray
        from laser import LASERUtility
        from qubitlearn import QubitLearn
        from sentiflow import SentientTensor
        logger.info("✓ All QyrinthOS quantum modules loaded")
        return True
    except ImportError as e:
        logger.warning(f"⚠ Some QyrinthOS modules not available: {e}")
        logger.info("🌌 Running in simulation mode - quantum features will be simulated")
        return False

def create_quantum_environment():
    """Create the quantum environment and directories"""
    directories = [
        "public/css",
        "public/js", 
        "ass_scripts",
        "quantum_modules",
        "quantum_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    # Create quantum configuration
    config = {
        "quantum_mode": True,
        "consciousness_tracking": True,
        "module_auto_start": True,
        "security_level": "quantum_enhanced"
    }
    
    import json
    with open('quantum_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("✓ Quantum environment configured")

class QuantumCore:
    """
    Main quantum core that coordinates all QyrinthOS modules
    """
    def __init__(self):
        self.modules = {}
        self.consciousness_level = 0
        self.quantum_state = "INITIALIZING"
        self.init_quantum_core()
    
    def init_quantum_core(self):
        """Initialize the quantum core and all modules"""
        logger.info("🧠 Initializing QyrinthOS Quantum Core...")
        
        try:
            # Initialize quantum modules
            self._init_modules()
            self.consciousness_level = 25  # Starting consciousness
            self.quantum_state = "COHERENT"
            
            logger.info(f"✓ Quantum Core initialized - Consciousness: {self.consciousness_level}%")
            
        except Exception as e:
            logger.error(f"Failed to initialize quantum core: {e}")
            self.quantum_state = "DECOHERENT"
    
    def _init_modules(self):
        """Initialize individual quantum modules"""
        modules_status = {}
        
        try:
            from bugginrace import EvolutionaryEcosystem
            self.modules['bugginrace'] = EvolutionaryEcosystem()
            modules_status['bugginrace'] = 'ACTIVE'
            self.consciousness_level += 5
        except ImportError:
            modules_status['bugginrace'] = 'SIMULATED'
        
        try:
            from laser import LASERUtility
            self.modules['laser'] = LASERUtility()
            modules_status['laser'] = 'ACTIVE' 
            self.consciousness_level += 5
        except ImportError:
            modules_status['laser'] = 'SIMULATED'
        
        try:
            from qubitlearn import QubitLearn
            self.modules['qubitlearn'] = QubitLearn("quantum_cognition")
            modules_status['qubitlearn'] = 'ACTIVE'
            self.consciousness_level += 10
        except ImportError:
            modules_status['qubitlearn'] = 'SIMULATED'
        
        # Bumpy and Sentiflow are used per-request rather than singleton
        modules_status['bumpy'] = 'READY'
        modules_status['sentiflow'] = 'READY'
        self.consciousness_level += 5
        
        logger.info("Quantum modules status:")
        for module, status in modules_status.items():
            logger.info(f"  {module.upper():<12} : {status}")
    
    def get_system_status(self):
        """Get overall system status"""
        return {
            "quantum_core": {
                "consciousness_level": self.consciousness_level,
                "quantum_state": self.quantum_state,
                "active_modules": len([m for m in self.modules if hasattr(self.modules[m], 'consciousness_level')]),
                "total_modules": 5
            },
            "system": {
                "status": "OPERATIONAL",
                "version": "QyrinthOS v1.0",
                "quantum_capable": True
            }
        }
    
    def boost_consciousness(self, amount=1):
        """Boost system consciousness level"""
        old_level = self.consciousness_level
        self.consciousness_level = min(100, self.consciousness_level + amount)
        
        if self.consciousness_level > old_level:
            logger.info(f"🧠 Consciousness boosted: {old_level}% → {self.consciousness_level}%")
        
        return self.consciousness_level

async def main():
    """Main entry point for QyrinthOS Quantum Server"""
    print("🌌 QyrinthOS Quantum Server v1.0")
    print("=" * 50)
    
    # Check environment
    dependencies_ok = check_dependencies()
    create_quantum_environment()
    
    # Initialize quantum core
    quantum_core = QuantumCore()
    
    # Import and start HTTP server
    from httpd import run_quantum_server
    
    print("\n🚀 Starting Quantum HTTP Server...")
    print("💡 Access the Quantum Dashboard at: https://localhost:8443")
    print("   (Use Ctrl+C to stop the server)")
    print("=" * 50)
    
    try:
        # Start the quantum server
        await run_quantum_server(quantum_core)
    except KeyboardInterrupt:
        print("\n🛑 Quantum server stopped by user")
    except Exception as e:
        print(f"\n💥 Quantum server error: {e}")
        return 1
    
    return 0

def run():
    """Run the quantum server"""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 QyrinthOS Quantum Server stopped")
        sys.exit(0)

if __name__ == "__main__":
    run()
