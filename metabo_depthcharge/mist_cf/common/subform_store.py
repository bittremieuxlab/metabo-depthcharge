"""Container storage for per-spectrum subformula assignments."""

import json
import os
import sqlite3
from pathlib import Path

import zstandard as zstd

FORMAT = "subform-store-1"
_SUFFIXES = (".subforms", ".subforms.sqlite")


# --------------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------------
class SubformStore:
    """Random access to a subformula container, keyed by spectrum name."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._pid = None
        self._conn = None
        self._dctx = None

    def _bind(self):
        # A DataLoader worker inherits this object across fork(). SQLite connections and zstd
        # contexts must not be shared over a fork, so (re)create them whenever the pid changes.
        if self._conn is not None and self._pid == os.getpid():
            return
        self._conn = sqlite3.connect(
            f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
        )
        self._pid = os.getpid()
        row = self._conn.execute("SELECT v FROM meta WHERE k='zdict'").fetchone()
        if row is not None and row[0]:
            self._dctx = zstd.ZstdDecompressor(dict_data=zstd.ZstdCompressionDict(row[0]))
        else:
            self._dctx = zstd.ZstdDecompressor()

    def __getitem__(self, name):
        self._bind()
        row = self._conn.execute(
            "SELECT data FROM blobs WHERE name=?", (str(name),)
        ).fetchone()
        if row is None:
            raise KeyError(name)
        return json.loads(self._dctx.decompress(row[0]))

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name):
        self._bind()
        return (
            self._conn.execute(
                "SELECT 1 FROM blobs WHERE name=? LIMIT 1", (str(name),)
            ).fetchone()
            is not None
        )

    def __len__(self):
        self._bind()
        return self._conn.execute("SELECT count(*) FROM blobs").fetchone()[0]

    def keys(self):
        self._bind()
        for (n,) in self._conn.execute("SELECT name FROM blobs"):
            yield n

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class LooseDirStore:
    """The legacy layout: one JSON per spectrum in a directory. Same interface as SubformStore."""

    def __init__(self, path):
        self.path = Path(path)

    def _p(self, name):
        return self.path / f"{name}.json"

    def __getitem__(self, name):
        p = self._p(name)
        if not p.exists():
            raise KeyError(name)
        with open(p) as fh:
            return json.load(fh)

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name):
        return self._p(name).exists()

    def keys(self):
        for f in os.scandir(self.path):
            if f.name.endswith(".json"):
                yield f.name[:-5]

    def close(self):
        pass


class ShardedSubformStore:
    """Random access across several SubformStore shards in one directory.

    Chunked/array-job producers (e.g. mist_cf.preprocessing.04_create_subformulae_assignment
    run over disjoint --start-idx/--end-idx slices) each own one shard file, so parallel
    writers never contend for the same SQLite file. This presents all shards as one store.
    """

    def __init__(self, shard_paths):
        self.shard_paths = sorted(shard_paths)
        self.path = self.shard_paths[0].parent
        self._pid = None
        self._stores = None
        self._index = None

    def _bind(self):
        if self._stores is not None and self._pid == os.getpid():
            return
        self._stores = [SubformStore(p) for p in self.shard_paths]
        self._index = {}
        for store in self._stores:
            for name in store.keys():
                self._index[name] = store
        self._pid = os.getpid()

    def __getitem__(self, name):
        self._bind()
        store = self._index.get(str(name))
        if store is None:
            raise KeyError(name)
        return store[name]

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name):
        self._bind()
        return str(name) in self._index

    def __len__(self):
        self._bind()
        return len(self._index)

    def keys(self):
        self._bind()
        return iter(self._index.keys())

    def close(self):
        if self._stores is not None:
            for s in self._stores:
                s.close()
        self._stores = None
        self._index = None


def open_subforms(path):
    """Open `path` as a container if it is one, else as a legacy directory.

    Also accepts a directory path whose sibling container exists, so callers can be pointed at
    either and keep working: `.../subformulae` finds `.../subformulae.subforms`. A directory
    holding multiple `*.subforms` shard files (chunked/array-job output) opens as one merged
    ShardedSubformStore; a directory of legacy loose JSONs falls back to LooseDirStore.
    """
    p = Path(path)
    if p.is_file():
        return SubformStore(p)
    for suf in _SUFFIXES:
        cand = Path(str(p) + suf)
        if cand.is_file():
            return SubformStore(cand)
    if p.is_dir():
        shards = [q for suf in _SUFFIXES for q in p.glob(f"*{suf}")]
        if shards:
            return ShardedSubformStore(shards)
        return LooseDirStore(p)
    raise FileNotFoundError(f"no subformula store or directory at {path}")


# --------------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------------
def train_dict(samples, dict_size=110 * 1024):
    """Train a shared zstd dictionary, or return b"" if there is too little to learn from."""
    if len(samples) < 8:
        return b""
    try:
        return zstd.train_dictionary(dict_size, samples).as_bytes()
    except zstd.ZstdError:
        return b""


def write_store(path, pairs, *, zdict=b"", columns=None, progress=None):
    """Build a container at `path` from an iterable of (name, already-compressed blob).

    Compression happens in the caller (a process pool during migration), because reading and
    parsing ~2M pretty-printed JSONs off NFS is the bottleneck, not the SQLite insert.

    `columns` is recorded in the metadata only; it describes what the caller projected. Leaving
    it None means the container is lossless, which is the migration default: the extra space over
    a slimmed store is small next to the overall saving, and it keeps the container a drop-in for
    any consumer, including ones that do not exist yet.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    conn = sqlite3.connect(tmp)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE meta(k TEXT PRIMARY KEY, v BLOB)")
    conn.execute("CREATE TABLE blobs(name TEXT PRIMARY KEY, data BLOB) WITHOUT ROWID")
    conn.executemany(
        "INSERT INTO meta VALUES(?,?)",
        [("format", FORMAT.encode()), ("zdict", zdict),
         ("columns", json.dumps(columns).encode())],
    )

    n = 0
    batch = []
    for name, blob in pairs:
        if blob is None:
            continue
        batch.append((str(name), blob))
        if len(batch) >= 2000:
            conn.executemany("INSERT INTO blobs VALUES(?,?)", batch)
            n += len(batch)
            batch = []
            if progress:
                progress(n)
    if batch:
        conn.executemany("INSERT INTO blobs VALUES(?,?)", batch)
        n += len(batch)
    conn.execute("INSERT INTO meta VALUES(?,?)", ("count", str(n).encode()))
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    os.replace(tmp, path)
    return n


def encode(obj, columns):
    """Object -> compact JSON bytes, optionally projected onto a subset of table columns."""
    return _encode(obj, columns)


def _encode(obj, columns):
    if columns is not None:
        obj = _project(obj, columns)
    return json.dumps(obj, separators=(",", ":")).encode()


def _project(obj, columns):
    """Keep only `columns` in every candidate table, for both on-disk schemas.

    mist-cf writes {formula: {ion: {"cand_tbl": tbl}}}; MIST/FLARE write
    {"cand_form":…, "cand_ion":…, "output_tbl": tbl}. A table is None when no subformula
    assignment was possible, and that None is meaningful -- it must survive.
    """
    keep = tuple(columns)

    def trim(tbl):
        return None if tbl is None else {k: tbl[k] for k in keep}

    if "output_tbl" in obj:
        out = dict(obj)
        out["output_tbl"] = trim(obj["output_tbl"])
        return out
    return {
        form: {ion: {"cand_tbl": trim(payload["cand_tbl"])} for ion, payload in ions.items()}
        for form, ions in obj.items()
    }
