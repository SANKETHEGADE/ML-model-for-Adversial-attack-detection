"""
blockchain.py

Lightweight blockchain used to certify ORIGINAL X-ray images.

Why this exists (separate from the CNN adversarial detector):
- The CNN in detector.py/app.py makes a STATISTICAL guess about whether an
  image looks perturbed. It can be fooled, especially by attacks like
  DeepFool that are designed to produce minimal, hard-to-notice perturbations.
- A blockchain hash check is DETERMINISTIC: if even a single pixel of a
  registered original X-ray is changed, its SHA-256 hash changes completely
  (avalanche effect), so tampering is caught with certainty -- not a
  probability -- as long as the original was registered on-chain first.

These two checks are complementary, not competing:
  CNN detector       -> "does this image LOOK adversarially perturbed?"
  Blockchain lookup   -> "does this EXACT image match a certified original?"

Design notes:
- Each block stores: patient_id, disease, image_hash (SHA-256 of the raw
  image bytes), timestamp, the previous block's hash, and its own hash.
- Proof-of-Work (low difficulty -- this is a tamper-evidence log for a
  single deployment, not a public/distributed chain) makes it
  computationally annoying to silently rewrite history.
- The chain is persisted to blockchain_data.json so records survive
  restarts. On load, the whole chain is re-validated (each block's stored
  hash is recomputed and compared, and previous_hash links are checked) --
  if anyone edits the JSON file by hand, load() detects it and refuses
  to start rather than silently trusting a tampered ledger.
"""

import hashlib
import json
import os
import time


class Block:
    def __init__(self, index, timestamp, patient_id, disease, image_hash,
                 previous_hash, nonce=0, hash_=None):
        self.index = index
        self.timestamp = timestamp
        self.patient_id = patient_id
        self.disease = disease
        self.image_hash = image_hash          
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = hash_ or self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "disease": self.disease,
            "image_hash": self.image_hash,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "disease": self.disease,
            "image_hash": self.image_hash,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash,
        }

    @staticmethod
    def from_dict(d):
        return Block(d["index"], d["timestamp"], d["patient_id"], d["disease"],
                     d["image_hash"], d["previous_hash"], d["nonce"], d["hash"])


DIFFICULTY = 4 
CHAIN_FILE = "blockchain_data.json"


class Blockchain:
    def __init__(self, chain_file=CHAIN_FILE):
        self.chain_file = chain_file
        self.chain = []
        self._load_or_create()

    
    def _load_or_create(self):
        if os.path.exists(self.chain_file):
            with open(self.chain_file, "r") as f:
                raw = json.load(f)
            self.chain = [Block.from_dict(b) for b in raw]
            valid, reason = self.is_chain_valid()
            if not valid:
                raise RuntimeError(
                    f"blockchain_data.json failed integrity check: {reason}. "
                    f"The file may have been edited outside the app. Refusing "
                    f"to start with a tampered ledger."
                )
        else:
            genesis = Block(0, time.time(), "GENESIS", "GENESIS", "0" * 64, "0")
            genesis = self._mine(genesis)
            self.chain = [genesis]
            self._save()

    def _save(self):
        with open(self.chain_file, "w") as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=2)

   
    def _mine(self, block):
        target = "0" * DIFFICULTY
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.compute_hash()
        return block

    def add_record(self, patient_id, disease, image_hash):
        previous = self.chain[-1]
        block = Block(
            index=previous.index + 1,
            timestamp=time.time(),
            patient_id=patient_id,
            disease=disease,
            image_hash=image_hash,
            previous_hash=previous.hash,
        )
        block = self._mine(block)
        self.chain.append(block)
        self._save()
        return block

    def find_by_image_hash(self, image_hash):
        """Returns the block certifying this exact image, or None if this
        exact image was never registered as an original."""
        for block in reversed(self.chain):  
            if block.image_hash == image_hash:
                return block
        return None

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current, previous = self.chain[i], self.chain[i - 1]
            if current.hash != current.compute_hash():
                return False, f"Block {current.index} hash doesn't match its own contents"
            if current.previous_hash != previous.hash:
                return False, f"Block {current.index} doesn't link to block {previous.index}"
            if not current.hash.startswith("0" * DIFFICULTY):
                return False, f"Block {current.index} doesn't satisfy proof-of-work"
        return True, "OK"


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
