#!/usr/bin/env python3
"""Prepare private episode packages, then explicitly upload or verify downloads."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.hf_archive import pack_episode, upload_package, download_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/hf_private_archive.json')
    commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('pack')
    pack.add_argument('--episode', type=Path, required=True)
    pack.add_argument('--cohort', required=True)
    pack.add_argument('--host', required=True)
    pack.add_argument('--stage', choices=('raw', 'rendered'), default='raw')
    pack.add_argument('--out', type=Path, default=ROOT / 'data/derived/hf_private_archive/staging')
    upload = commands.add_parser('upload')
    upload.add_argument('--package', type=Path, required=True)
    upload.add_argument('--run', action='store_true', help='commit this completed archive to the private Dataset')
    download = commands.add_parser('download')
    download.add_argument('--prefix', required=True)
    download.add_argument('--destination', type=Path, required=True)
    download.add_argument('--revision')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == 'pack':
        path, manifest = pack_episode(args.episode, cohort=args.cohort, host=args.host, stage=args.stage, output=args.out)
        print('PACKAGE_READY ' + str(path))
        print('PAYLOAD bytes=' + str(manifest['payload']['bytes']) + ' sha256=' + manifest['payload']['sha256'])
    elif args.command == 'upload':
        if not args.run:
            m = json.loads((args.package / 'manifest.json').read_text())
            print('UPLOAD_PLAN repo=' + config['repo_id'] + ' private_required=true prefix=' + m['prefix'] + ' bytes=' + str(m['payload']['bytes']))
            print('PREPARED_ONLY cloud_write=false')
        else:
            print('UPLOAD_COMPLETED ' + json.dumps(upload_package(args.package, config)))
    else:
        print('DOWNLOAD_VERIFIED ' + json.dumps(download_package(args.prefix, config, args.destination, revision=args.revision)))


if __name__ == '__main__':
    main()
