"""Proof Chain Module for DiademAce-v11-Arbiter

Implements cryptographic proof chain verification and validation.
Full audit trail with immutable logging.
"""

import hashlib
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger('DiademAce.ProofChain')


@dataclass
class ProofBlock:
    """Individual proof block in the chain"""
    index: int
    timestamp: str
    data: Dict
    previous_hash: str
    hash: str
    confidence: float
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ProofChain:
    """Immutable proof chain for verification auditability"""
    
    def __init__(self, chain_id: str = None):
        self.chain_id = chain_id or self._generate_chain_id()
        self.chain: List[ProofBlock] = []
        self.audit_log: List[Dict] = []
        
        # Create genesis block
        self._create_genesis_block()
        
        logger.info(f"ProofChain initialized: {self.chain_id}")
    
    def _generate_chain_id(self) -> str:
        """Generate unique chain ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_data = {
            'type': 'genesis',
            'chain_id': self.chain_id,
            'timestamp': datetime.now().isoformat()
        }
        
        genesis_hash = self._calculate_hash(0, genesis_data, '0' * 64)
        
        genesis_block = ProofBlock(
            index=0,
            timestamp=datetime.now().isoformat(),
            data=genesis_data,
            previous_hash='0' * 64,
            hash=genesis_hash,
            confidence=1.0,
            metadata={'genesis': True}
        )
        
        self.chain.append(genesis_block)
        self._log_audit('genesis_block_created', genesis_block.to_dict())
    
    def _calculate_hash(self, index: int, data: Dict, previous_hash: str) -> str:
        """Calculate cryptographic hash for a block"""
        block_string = json.dumps({
            'index': index,
            'data': data,
            'previous_hash': previous_hash
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def add_proof(
        self,
        proof_data: Dict,
        confidence: float,
        metadata: Dict = None
    ) -> ProofBlock:
        """Add a new proof to the chain"""
        
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        previous_block = self.chain[-1]
        new_index = len(self.chain)
        timestamp = datetime.now().isoformat()
        
        # Calculate hash
        new_hash = self._calculate_hash(new_index, proof_data, previous_block.hash)
        
        # Create new proof block
        new_block = ProofBlock(
            index=new_index,
            timestamp=timestamp,
            data=proof_data,
            previous_hash=previous_block.hash,
            hash=new_hash,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Validate before adding
        if self._validate_block(new_block, previous_block):
            self.chain.append(new_block)
            self._log_audit('proof_added', {
                'block_index': new_index,
                'hash': new_hash,
                'confidence': confidence
            })
            logger.info(f"Proof block added: index={new_index}, confidence={confidence:.4f}")
            return new_block
        else:
            raise ValueError("Invalid proof block")
    
    def _validate_block(self, block: ProofBlock, previous_block: ProofBlock) -> bool:
        """Validate a proof block"""
        
        # Check index
        if block.index != previous_block.index + 1:
            logger.error(f"Invalid block index: expected {previous_block.index + 1}, got {block.index}")
            return False
        
        # Check previous hash
        if block.previous_hash != previous_block.hash:
            logger.error("Previous hash mismatch")
            return False
        
        # Recalculate hash
        calculated_hash = self._calculate_hash(block.index, block.data, block.previous_hash)
        if block.hash != calculated_hash:
            logger.error("Hash verification failed")
            return False
        
        return True
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify the entire proof chain integrity"""
        errors = []
        
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            if not self._validate_block(current, previous):
                errors.append(f"Invalid block at index {i}")
        
        is_valid = len(errors) == 0
        
        self._log_audit('chain_verification', {
            'valid': is_valid,
            'total_blocks': len(self.chain),
            'errors': errors
        })
        
        return is_valid, errors
    
    def get_chain_confidence(self) -> float:
        """Calculate overall chain confidence"""
        if len(self.chain) <= 1:
            return 1.0
        
        # Weighted average of confidence scores
        confidences = [block.confidence for block in self.chain[1:]]  # Skip genesis
        return sum(confidences) / len(confidences)
    
    def _log_audit(self, event: str, data: Dict):
        """Log audit event"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'data': data,
            'chain_id': self.chain_id
        }
        self.audit_log.append(audit_entry)
    
    def export_chain(self, filepath: str):
        """Export proof chain to JSON file"""
        export_data = {
            'chain_id': self.chain_id,
            'created': self.chain[0].timestamp if self.chain else None,
            'blocks': [block.to_dict() for block in self.chain],
            'audit_log': self.audit_log,
            'chain_confidence': self.get_chain_confidence(),
            'exported': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Proof chain exported to {filepath}")
    
    def get_summary(self) -> Dict:
        """Get chain summary for audit reports"""
        is_valid, errors = self.verify_chain()
        
        return {
            'chain_id': self.chain_id,
            'total_blocks': len(self.chain),
            'valid': is_valid,
            'errors': errors,
            'average_confidence': self.get_chain_confidence(),
            'latest_block': self.chain[-1].to_dict() if self.chain else None
        }


def create_proof_chain(chain_id: str = None) -> ProofChain:
    """Factory function to create a new proof chain"""
    return ProofChain(chain_id=chain_id)


if __name__ == "__main__":
    # Test proof chain
    logging.basicConfig(level=logging.INFO)
    
    chain = create_proof_chain()
    
    # Add test proofs
    chain.add_proof(
        proof_data={'test': 'proof_1', 'value': 42},
        confidence=0.95,
        metadata={'source': 'test'}
    )
    
    chain.add_proof(
        proof_data={'test': 'proof_2', 'value': 84},
        confidence=0.87,
        metadata={'source': 'test'}
    )
    
    # Verify chain
    is_valid, errors = chain.verify_chain()
    print(f"Chain valid: {is_valid}")
    print(f"Chain confidence: {chain.get_chain_confidence():.4f}")
    print(f"Total blocks: {len(chain.chain)}")
    
    # Get summary
    summary = chain.get_summary()
    print(json.dumps(summary, indent=2))
