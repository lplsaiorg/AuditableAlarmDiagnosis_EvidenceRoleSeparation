"""Cross-process, cross-model admission and cooldown for all project API calls."""
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import os
from pathlib import Path
import sqlite3
import time
import uuid

DEFAULT_DB=Path(__file__).resolve().parents[1]/'tmp/api_control/minimax_api.sqlite3'


def alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5
        try:
            code=wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True


class SharedGate:
    def __init__(self,path=None,clock=time.time,sleep=time.sleep):
        self.path=Path(path or os.environ.get('PAPER3_API_GATE_DB',DEFAULT_DB))
        self.path.parent.mkdir(parents=True,exist_ok=True);self.clock=clock;self.sleep=sleep
        with self.db() as c:
            c.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), next_start REAL, cooldown REAL, gap REAL, streak INTEGER, last_429 REAL)')
            c.execute('INSERT OR IGNORE INTO state VALUES (1,0,0,5,0,0)')
            c.execute('CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, pid INTEGER, expires REAL)')
            c.execute('CREATE TABLE IF NOT EXISTS events (at REAL, kind TEXT, detail TEXT)')

    @contextmanager
    def db(self):
        c=sqlite3.connect(self.path,timeout=30,isolation_level=None)
        try:
            c.execute('BEGIN IMMEDIATE');yield c;c.execute('COMMIT')
        except BaseException:
            c.execute('ROLLBACK');raise
        finally:c.close()

    def try_acquire(self):
        now=self.clock()
        with self.db() as c:
            # HTTP read timeout is <=300s. Leases remain conservatively held for
            # 600s after process death, without allowing a fifth live request.
            for ident,pid in c.execute('SELECT id,pid FROM leases WHERE expires<=?',(now,)).fetchall():
                if not alive(pid):c.execute('DELETE FROM leases WHERE id=?',(ident,))
            next_start,cooldown,gap,streak,last=c.execute('SELECT next_start,cooldown,gap,streak,last_429 FROM state WHERE id=1').fetchone()
            wait=max(next_start,cooldown)-now
            if wait>0:return None,min(wait,1.0)
            if c.execute('SELECT COUNT(*) FROM leases').fetchone()[0]>=4:return None,.25
            ident=uuid.uuid4().hex
            c.execute('INSERT INTO leases VALUES (?,?,?)',(ident,os.getpid(),now+600))
            c.execute('UPDATE state SET next_start=? WHERE id=1',(now+gap,))
            c.execute('INSERT INTO events VALUES (?,?,?)',(now,'admitted',ident))
            return ident,0

    def acquire(self):
        while True:
            ident,wait=self.try_acquire()
            if ident:return ident
            self.sleep(wait)

    def release(self,ident):
        with self.db() as c:c.execute('DELETE FROM leases WHERE id=?',(ident,))

    def rate_limited(self,retry_after=None):
        now=self.clock();server_wait=0
        if retry_after:
            try:server_wait=max(0,float(retry_after))
            except (TypeError,ValueError):
                try:server_wait=max(0,parsedate_to_datetime(retry_after).timestamp()-now)
                except (TypeError,ValueError,OverflowError):pass
        with self.db() as c:
            cooldown,gap,streak,last=c.execute('SELECT cooldown,gap,streak,last_429 FROM state WHERE id=1').fetchone()
            streak=streak+1 if now-last<3600 else 1
            delay=max(server_wait,min(900,60*2**min(streak-1,4)))
            until=max(cooldown,now+delay)
            c.execute('UPDATE state SET cooldown=?,gap=?,streak=?,last_429=? WHERE id=1',(until,min(60,gap*2),streak,now))
            c.execute('INSERT INTO events VALUES (?,?,?)',(now,'429_cooldown',str(until)))
        return until

    def snapshot(self):
        with self.db() as c:
            row=c.execute('SELECT next_start,cooldown,gap,streak,last_429 FROM state WHERE id=1').fetchone()
            active=c.execute('SELECT COUNT(*) FROM leases WHERE expires>?',(self.clock(),)).fetchone()[0]
        return dict(zip(['next_start','cooldown_until','minimum_interval_seconds','rate_limit_streak','last_429'],row),active_requests=active,max_concurrency=4)

    def overloaded(self,retry_after=None):
        now=self.clock()
        try:server_wait=max(0,float(retry_after or 0))
        except (ValueError,TypeError):server_wait=0
        with self.db() as c:
            count=c.execute("SELECT COUNT(*) FROM events WHERE kind='529_cooldown' AND at>?",(now-1800,)).fetchone()[0]
            until=max(c.execute('SELECT cooldown FROM state WHERE id=1').fetchone()[0],now+max(server_wait,min(300,30*2**min(count,4))))
            c.execute('UPDATE state SET cooldown=? WHERE id=1',(until,))
            c.execute('INSERT INTO events VALUES (?,?,?)',(now,'529_cooldown',str(until)))
        return until
