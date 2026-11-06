"""Per-credential cross-process leases and isolated cooldowns, round-robin routing."""
import hashlib
from pathlib import Path
import time

from .admission import LegacySlots
from .transport import DirectTransport
from shared_api_gate import DEFAULT_DB


class KeyPool:
    def __init__(self, keys, *, control_dir=None, transport_factory=DirectTransport):
        if not 1 <= len(keys) <= 3 or len(set(keys)) != len(keys):
            raise ValueError('Provide one to three distinct credentials')
        directory=Path(control_dir or DEFAULT_DB.parent)
        self.labels=[hashlib.sha256(key.encode()).hexdigest()[:16] for key in keys]
        # Primary shares the existing gate with single-key callers; callers keep
        # the vault primary in position zero. Additional credentials use fingerprints.
        self.gates=[LegacySlots(directory/(DEFAULT_DB.name if i==0 else 'key_'+label+'.sqlite3'),max_concurrency=6)
                    for i,label in enumerate(self.labels)]
        self.transports=[transport_factory(key) for key in keys]
        self.max_concurrency=6*len(keys)
        self.cursor=0

    def try_acquire(self):
        # Dispatcher holds its condition lock; SQLite serializes other processes.
        for offset in range(len(self.gates)):
            index=(self.cursor+offset)%len(self.gates)
            lease=self.gates[index].try_acquire()
            if lease is not None:
                self.cursor=(index+1)%len(self.gates)
                return index,lease
        return None

    def release(self, lease):
        index,ident=lease
        self.gates[index].release(ident)

    def dispatch(self, lease, path, payload):
        return self.transports[lease[0]](path,payload)

    def rejected(self, lease, status, delay):
        gate=self.gates[lease[0]].gate
        if status==429:gate.rate_limited(str(delay))
        else:gate.overloaded(str(delay))

    def label(self, lease):
        return self.labels[lease[0]]

    def snapshot(self):
        result=[]
        for label,slot in zip(self.labels,self.gates):
            with slot.gate.db() as c:
                count=c.execute('SELECT COUNT(*) FROM leases').fetchone()[0]
                until=c.execute('SELECT cooldown FROM state WHERE id=1').fetchone()[0]
            result.append({'key_id':label,'active':count,'limit':6,'cooldown_until':until})
        return result

    def close(self):
        for transport in self.transports:transport.close()
