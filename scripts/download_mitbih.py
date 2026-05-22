"""Téléchargement de la MIT-BIH Arrhythmia Database via WFDB.

Usage:
    python scripts/download_mitbih.py --dest data/mitbih
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.mitbih_loader import download_mitbih


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=str, default='data/mitbih', help='Destination directory')
    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    try:
        download_mitbih(args.dest)
    except Exception as e:
        print('Download failed:', e)


if __name__ == '__main__':
    main()
