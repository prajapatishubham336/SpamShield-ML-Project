import csv
import sys
from pathlib import Path

# Add project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import RAW_DATA_DIR


def make_sample_data() -> Path:
    output_path = RAW_DATA_DIR / "spam.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = [
        ("Hey, are we meeting for lunch today?", "ham"),
        ("Please send me the class notes.", "ham"),
        ("Your appointment is confirmed for tomorrow.", "ham"),
        ("Can you call me when you are free?", "ham"),
        ("Thanks for helping me with the assignment.", "ham"),
        ("Are you coming to the meeting?", "ham"),
        ("Your package will arrive this afternoon.", "ham"),
        ("Please review the attached project report.", "ham"),
        ("I will reach the station in ten minutes.", "ham"),
        ("Happy birthday! Have a wonderful day.", "ham"),
        ("Congratulations! You won a free prize. Claim now!", "spam"),
        ("URGENT! Claim your cash reward immediately!", "spam"),
        ("You have won a gift card. Click to claim.", "spam"),
        ("Exclusive offer! Get a discount today.", "spam"),
        ("Win a free vacation. Reply now to enter.", "spam"),
        ("Your account won a bonus. Claim your reward.", "spam"),
        ("Limited offer! Buy now and save big.", "spam"),
        ("You are selected for a cash prize. Call now.", "spam"),
        ("Free entry in our prize draw. Text to join.", "spam"),
        ("Act now to claim your special reward.", "spam"),
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])

        for _ in range(10):
            writer.writerows(examples)

    print(f"Demo dataset created successfully: {output_path}")
    print(f"Total data rows: {len(examples) * 10}")
    print("Note: This is demo data, not a production-quality dataset.")

    return output_path


if __name__ == "__main__":
    make_sample_data()
