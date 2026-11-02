"""Fetch only the pinned official code/config/tokenizer; no weight downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(destination):
    manifest = json.loads(Path(__file__).with_name('v41_reference_sources.json').read_text())
    root = f'https://huggingface.co/{manifest["repo"]}/resolve/{manifest["revision"]}'
    for name, entry in manifest['files'].items():
        path = destination / name
        if path.exists():
            content = path.read_bytes()
        else:
            content = urllib.request.urlopen(f'{root}/{name}', timeout=60).read()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError(f'source mismatch: {name}; existing files are never overwritten')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(content)
        print(f'verified {name}')
    (destination / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('destination', type=Path)
    fetch(ap.parse_args().destination)
