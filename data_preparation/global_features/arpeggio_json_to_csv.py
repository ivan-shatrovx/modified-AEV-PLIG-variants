#!/usr/bin/env python3
"""
Convert Arpeggio JSON output to CSV summary with interaction counts.

This script parses the JSON output from pdbe-arpeggio and generates a CSV
file summarizing the number of each interaction type, similar to the 
Arpeggio web interface summary.

Usage:
    python arpeggio_json_to_csv.py input.json -o summary.csv
    python arpeggio_json_to_csv.py input.json -o summary.csv --ligand-only
"""

import argparse
import json
import csv
import os
from collections import defaultdict
from pathlib import Path


def parse_arpeggio_json(json_path: str, ligand_only: bool = True, 
                        ligand_resname: str = "LIG") -> dict:
    """
    Parse Arpeggio JSON and count interactions.
    
    Args:
        json_path: Path to Arpeggio JSON output
        ligand_only: If True, only count interactions involving the ligand
        ligand_resname: Residue name of the ligand
    
    Returns:
        Dictionary with interaction counts
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize counters
    counts = defaultdict(int)
    
    # Track unique interactions for mutually exclusive counting
    atom_pairs_seen = set()
    
    for interaction in data:
        # Check if interaction involves ligand
        if ligand_only:
            bgn_res = interaction.get('bgn', {}).get('label_comp_id', '')
            end_res = interaction.get('end', {}).get('label_comp_id', '')
            
            if bgn_res != ligand_resname and end_res != ligand_resname:
                continue
        
        # Get contact types
        contacts = interaction.get('contact', [])
        interaction_type = interaction.get('type', 'atom-atom')
        
        # Create unique identifier for this atom pair
        bgn = interaction.get('bgn', {})
        end = interaction.get('end', {})
        
        if interaction_type == 'atom-atom':
            pair_id = (
                f"{bgn.get('auth_asym_id')}:{bgn.get('auth_seq_id')}:{bgn.get('auth_atom_id')}",
                f"{end.get('auth_asym_id')}:{end.get('auth_seq_id')}:{end.get('auth_atom_id')}"
            )
            pair_id = tuple(sorted(pair_id))
            
            if pair_id not in atom_pairs_seen:
                atom_pairs_seen.add(pair_id)
                counts['total_contacts'] += 1
        
        # Count each contact type
        for contact in contacts:
            counts[contact] += 1
    
    return dict(counts)


def generate_summary_csv(counts: dict, output_path: str, name: str = ""):
    """
    Generate a CSV summary in the Arpeggio web format.
    
    Args:
        counts: Dictionary with interaction counts
        output_path: Path for output CSV
        name: Optional name/identifier for this complex
    """
    
    # Define the categories matching the Arpeggio web interface
    categories = {
        # Mutually Exclusive Interactions
        'Mutually Exclusive Interactions': [
            ('Total number of contacts', 'total_contacts'),
            ('Of which VdW interactions', 'vdw'),
            ('Of which VdW clash interactions', 'vdw_clash'),
            ('Of which covalent interactions', 'covalent'),
            ('Of which covalent clash interactions', 'covalent_clash'),
            ('Of which proximal', 'proximal'),
        ],
        # Polar Contacts
        'Polar Contacts': [
            ('Polar contacts', 'polar'),
            ('Water mediated polar contacts', 'water_polar'),
            ('Weak polar contacts', 'weak_polar'),
            ('Water mediated weak polar contacts', 'water_weak_polar'),
        ],
        # Feature Contacts
        'Feature Contacts': [
            ('Hydrogen bonds', 'hbond'),
            ('Water mediated hydrogen bonds', 'water_hbond'),
            ('Weak hydrogen bonds', 'weak_hbond'),
            ('Water mediated weak hydrogen bonds', 'water_weak_hbond'),
            ('Halogen bonds', 'xbond'),
            ('Ionic interactions', 'ionic'),
            ('Metal complex interactions', 'metal'),
            ('Aromatic contacts', 'aromatic'),
            ('Hydrophobic contacts', 'hydrophobic'),
            ('Carbonyl interactions', 'carbonyl'),
        ],
        # Pi interactions (from plane interactions)
        'Pi Interactions': [
            ('Carbon-Pi', 'CARBONPI'),
            ('Cation-Pi', 'CATIONPI'),
            ('Donor-Pi', 'DONORPI'),
            ('Halogen-Pi', 'HALOGENPI'),
            ('Sulphur-Pi', 'METSULPHURPI'),
        ],
        # Ring interactions
        'Ring Interactions': [
            ('Face-to-face (FF)', 'FF'),
            ('Face-to-edge (FE)', 'FE'),
            ('Edge-to-face (EF)', 'EF'),
            ('Edge-to-edge (EE)', 'EE'),
            ('Offset face-to-face (OF)', 'OF'),
            ('Offset edge-to-face (OE)', 'OE'),
            ('Face-to-tilted (FT)', 'FT'),
            ('Edge-to-tilted (ET)', 'ET'),
        ],
        # Group interactions
        'Group Interactions': [
            ('Amide-amide', 'AMIDEAMIDE'),
            ('Amide-ring', 'AMIDERING'),
        ],
    }
    
    rows = []
    
    # Add name column if provided
    if name:
        rows.append(['Complex', name])
        rows.append([])
    
    for category, items in categories.items():
        rows.append([category, 'Count'])
        for label, key in items:
            count = counts.get(key, 0)
            rows.append([label, count])
        rows.append([])  # Empty row between categories
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    return rows


def generate_flat_csv(counts: dict, output_path: str, name: str = ""):
    """
    Generate a flat CSV with one row per complex (good for batch processing).
    
    Args:
        counts: Dictionary with interaction counts
        output_path: Path for output CSV
        name: Optional name/identifier for this complex
    """
    
    # All possible columns in a logical order
    columns = [
        'name',
        'total_contacts',
        'vdw', 'vdw_clash', 'covalent', 'covalent_clash', 'proximal',
        'polar', 'weak_polar',
        'hbond', 'weak_hbond', 'xbond',
        'ionic', 'metal', 'aromatic', 'hydrophobic', 'carbonyl',
        'CARBONPI', 'CATIONPI', 'DONORPI', 'HALOGENPI', 'METSULPHURPI',
        'FF', 'FE', 'EF', 'EE', 'OF', 'OE', 'FT', 'ET',
        'AMIDEAMIDE', 'AMIDERING'
    ]
    
    row = {'name': name}
    for col in columns[1:]:
        row[col] = counts.get(col, 0)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    
    return row


def batch_to_csv(json_files: list, output_path: str, 
                 ligand_only: bool = True, ligand_resname: str = "LIG"):
    """
    Process multiple JSON files and create a combined CSV.
    
    Args:
        json_files: List of paths to Arpeggio JSON files
        output_path: Path for output CSV
        ligand_only: If True, only count interactions involving the ligand
        ligand_resname: Residue name of the ligand
    """
    
    columns = [
        'name',
        'total_contacts',
        'vdw', 'vdw_clash', 'covalent', 'covalent_clash', 'proximal',
        'polar', 'weak_polar',
        'hbond', 'weak_hbond', 'xbond',
        'ionic', 'metal', 'aromatic', 'hydrophobic', 'carbonyl',
        'CARBONPI', 'CATIONPI', 'DONORPI', 'HALOGENPI', 'METSULPHURPI',
        'FF', 'FE', 'EF', 'EE', 'OF', 'OE', 'FT', 'ET',
        'AMIDEAMIDE', 'AMIDERING'
    ]
    
    rows = []
    
    for json_file in json_files:
        name = Path(json_file).stem
        counts = parse_arpeggio_json(json_file, ligand_only, ligand_resname)
        
        row = {'name': name}
        for col in columns[1:]:
            row[col] = counts.get(col, 0)
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Convert Arpeggio JSON output to CSV summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file - formatted summary
    %(prog)s complex.json -o summary.csv
    
    # Single file - flat format (one row)
    %(prog)s complex.json -o summary.csv --flat
    
    # Multiple files - combined CSV
    %(prog)s *.json -o batch_summary.csv --batch
    
    # Only count ligand interactions
    %(prog)s complex.json -o summary.csv --ligand-only --resname LIG
        """
    )
    
    parser.add_argument('input', nargs='+', help='Input Arpeggio JSON file(s)')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file')
    parser.add_argument('--flat', action='store_true',
                        help='Generate flat CSV (one row per complex)')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: combine multiple JSON files into one CSV')
    parser.add_argument('--ligand-only', action='store_true', default=True,
                        help='Only count interactions involving the ligand (default: True)')
    parser.add_argument('--all-interactions', action='store_true',
                        help='Count all interactions, not just ligand ones')
    parser.add_argument('--resname', default='LIG',
                        help='Ligand residue name (default: LIG)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    ligand_only = not args.all_interactions
    
    if args.batch or len(args.input) > 1:
        # Batch mode
        rows = batch_to_csv(args.input, args.output, ligand_only, args.resname)
        print(f"Processed {len(rows)} files -> {args.output}")
        
    else:
        # Single file
        input_file = args.input[0]
        name = Path(input_file).stem
        
        counts = parse_arpeggio_json(input_file, ligand_only, args.resname)
        
        if args.verbose:
            print(f"Interaction counts for {name}:")
            for k, v in sorted(counts.items()):
                if v > 0:
                    print(f"  {k}: {v}")
        
        if args.flat:
            generate_flat_csv(counts, args.output, name)
        else:
            generate_summary_csv(counts, args.output, name)
        
        print(f"Summary written to: {args.output}")


if __name__ == '__main__':
    main()
