"""Reproducible, explicitly synthetic demo corpus; no external or personal data."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random

from src.data_loader import ROOT

VERSION = 1
# These are design parameters, not claims about observed student expenditure.
COUNTS = {
    'Food and Dining': 360, 'Groceries': 280, 'Transportation': 320,
    'Shopping': 240, 'Entertainment': 180, 'Bills and Utilities': 200,
    'Health': 160, 'Travel': 160, 'Subscriptions': 180, 'Education': 240,
}
ITEMS = {
    'Food and Dining': ['cafe lunch', 'canteen breakfast', 'coffee and sandwich', 'dinner thali', 'pizza takeaway', 'tea and samosa', 'dosa meal', 'burger combo', 'restaurant bill', 'noodles bowl', 'juice and snacks', 'mess meal', 'lunch delivery', 'bakery pastry', 'biryani dinner', 'campus cafe coffee'],
    'Groceries': ['rice and lentils', 'weekly vegetables', 'milk and bread', 'eggs and butter', 'fresh fruit', 'cooking oil', 'flour and sugar', 'salt and spices', 'oats and cereal', 'grocery basket', 'tomatoes and onions', 'curd and paneer', 'pasta and beans', 'frozen peas', 'potatoes and carrots', 'kitchen staples'],
    'Transportation': ['city bus ticket', 'metro recharge', 'auto rickshaw fare', 'local cab ride', 'campus shuttle', 'bike taxi fare', 'bus pass', 'city train fare', 'bicycle rental', 'scooter petrol', 'parking fee', 'metro ride', 'shared auto fare', 'daily commute', 'local transit pass', 'ride to campus'],
    'Shopping': ['cotton t shirt', 'new jeans', 'running shoes', 'phone cover', 'headphones', 'backpack', 'desk lamp', 'water bottle', 'sweatshirt', 'usb charger', 'laundry detergent', 'umbrella', 'bedsheet', 'socks', 'laptop stand', 'clothes purchase'],
    'Entertainment': ['cinema ticket', 'concert entry', 'bowling game', 'arcade tokens', 'theatre show', 'comedy night', 'music festival', 'museum admission', 'amusement park', 'gaming session', 'movie screening', 'karaoke entry', 'dance event', 'cricket match entry', 'escape room', 'live gig'],
    'Bills and Utilities': ['electricity bill', 'water bill', 'gas cylinder refill', 'mobile recharge', 'broadband bill', 'internet connection fee', 'hostel electricity charge', 'phone bill', 'prepaid data pack', 'utility payment', 'wifi bill', 'gas bill', 'sewage charge', 'power meter recharge', 'municipal water charge', 'postpaid mobile bill'],
    'Health': ['pharmacy medicine', 'doctor consultation', 'dental checkup', 'blood test', 'first aid supplies', 'eye examination', 'prescription tablets', 'clinic appointment', 'physiotherapy session', 'vaccination fee', 'cough syrup', 'vitamin supplements', 'medical lab test', 'bandages and antiseptic', 'health checkup', 'allergy medication'],
    'Travel': ['intercity train booking', 'flight ticket', 'hotel room', 'weekend hostel stay', 'outstation bus booking', 'holiday tour', 'airport transfer', 'trip accommodation', 'guesthouse booking', 'travel insurance', 'sleeper coach reservation', 'railway reservation', 'overnight journey ticket', 'luggage storage on trip', 'camping stay', 'interstate coach ticket'],
    'Subscriptions': ['monthly music plan', 'video streaming membership', 'cloud storage renewal', 'software subscription', 'podcast premium plan', 'annual streaming renewal', 'digital newspaper plan', 'audiobook membership', 'movie platform renewal', 'online game membership', 'password manager plan', 'photo editor renewal', 'ad free music renewal', 'monthly magazine plan', 'coding tool subscription', 'monthly entertainment plan'],
    'Education': ['textbook purchase', 'notebook and pens', 'photocopy notes', 'exam registration', 'course enrollment', 'lab manual', 'printing assignment', 'tuition fee', 'workshop registration', 'library fine', 'reference book', 'lab equipment fee', 'academic journal copy', 'study material', 'semester registration', 'stationery supplies'],
}
DEFINITIONS = {
    'Food and Dining': 'Prepared meals, drinks and food service.',
    'Groceries': 'Ingredients and household food supplies for preparation.',
    'Transportation': 'Local commuting, local transit, fuel and parking.',
    'Shopping': 'Clothing, electronics and non-food household goods.',
    'Entertainment': 'One-time leisure activities and event admission.',
    'Bills and Utilities': 'Electricity, water, gas, telecom and connectivity charges.',
    'Health': 'Medical care, pharmacy products and diagnostic services.',
    'Travel': 'Intercity journeys, accommodation and trip services.',
    'Subscriptions': 'Recurring digital content, membership and software plans.',
    'Education': 'Academic fees, study materials, printing and stationery.',
}
MERCHANTS = ['cedar corner', 'amber plaza', 'willow junction', 'maple square',
             'cobalt point', 'meadow hub', 'silver avenue', 'coral courtyard',
             'birch centre', 'indigo place', 'marigold outlet', 'orchid lane',
             'pebble mart', 'harbor kiosk', 'saffron stop', 'fern house']
TEMPLATES = ['UPI payment at {merchant} for {item}', '{merchant} - {item}',
             'paid for {item} at {merchant}', 'card purchase {merchant} {item}',
             '{item} via {merchant}', 'cash spent on {item} {merchant}',
             'campus purchase: {item} from {merchant}', 'debit {merchant} {item}',
             'payment to {merchant}: {item}', 'purchase of {item} at {merchant}',
             '{merchant} receipt for {item}', 'online payment {item} {merchant}']


def generate_records(seed=42, counts=None):
    rng = random.Random(seed)
    counts = COUNTS if counts is None else counts
    rows, ambiguous_groups = [], 0
    for category, count in counts.items():
        if category not in ITEMS or count < 10 or count % 2:
            raise ValueError('Use documented categories and even counts of at least 10.')
        seen = set()
        for _ in range(count // 2):
            # Deliberately include descriptions whose true simulated purpose is
            # absent. This is realistic ambiguity, not independent human labeling.
            ambiguous = rng.random() < 0.08
            for attempt in range(10000):
                merchant = rng.choice(MERCHANTS)
                item = rng.choice(ITEMS[category]) if not ambiguous else rng.choice(['purchase', 'payment', 'order', 'receipt'])
                base = rng.choice(TEMPLATES).format(merchant=merchant, item=item)
                if base.casefold() not in seen:
                    break
            else:
                raise ValueError('Requested corpus exceeds the available distinct template combinations.')
            seen.add(base.casefold())
            ambiguous_groups += int(ambiguous)
            style = rng.random()
            if style < 0.12:
                base = base.upper()
            elif style < 0.30:
                base = base.lower()
            # Numeric variants deliberately expose the weakness of a random
            # exact-description split. Grouped evaluation removes this overlap.
            for reference in rng.sample(range(100000, 1000000), 2):
                rows.append({'description': f'{base} ref {reference}', 'category': category})
    rng.shuffle(rows)
    return rows, {'ambiguous_groups': ambiguous_groups, 'base_groups': len(rows) // 2,
                  'rows': len(rows), 'seed': seed, 'generator_version': VERSION}


def write_dataset(root=ROOT, seed=42, activate=False):
    root = Path(root)
    rows, generation = generate_records(seed)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['description', 'category'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode('utf-8')
    filename = f'spendsense_synthetic_v{VERSION}_seed{seed}.csv'
    raw = root / 'data' / 'raw' / filename
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists() and raw.read_bytes() != content:
        raise ValueError('Existing generated file differs. Preserve it and bump the generator version before changing the recipe.')
    raw.write_bytes(content)
    manifest = {
        'verified': True, 'file': f'raw/{filename}',
        'description_column': 'description', 'target_column': 'category',
        'source_url': 'local:src/generate_synthetic.py',
        'license': 'Project-authored synthetic demonstration data; authorized for this project',
        'license_evidence': 'docs/synthetic_dataset.md — authorship and requested use',
        'provenance': 'synthetic',
        'provenance_notes': 'Generated locally at the user\'s explicit request. No bank/customer data, external dataset, or Kaggle labels were used. Labels follow the authored generation recipe, not independent annotation.',
        'taxonomy': list(COUNTS), 'taxonomy_evidence': 'docs/synthetic_dataset.md and DEFINITIONS in src/generate_synthetic.py',
        'feature_definitions': {'description': 'Fictional transaction text; only model input.',
                                'category': 'Single label assigned by the documented synthetic generation scenario; prediction target only.'},
        'sha256': hashlib.sha256(content).hexdigest(), 'generation': generation,
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'category_definitions': DEFINITIONS,
    }
    path = root / 'data' / 'synthetic.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if activate:
        (root / 'data' / 'dataset.json').write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
    return path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--activate', action='store_true', help='Explicitly set this source as data/dataset.json.')
    args = parser.parse_args()
    path, manifest = write_dataset(seed=args.seed, activate=args.activate)
    print(json.dumps({'manifest': str(path), **manifest['generation'], 'sha256': manifest['sha256'], 'activated': args.activate}, indent=2))


if __name__ == '__main__':
    main()
