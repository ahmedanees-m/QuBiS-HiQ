import json
import os
import sys


def validate_json_files():
    errors = []
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(root, f)
                try:
                    with open(path) as fp:
                        json.load(fp)
                except json.JSONDecodeError as e:
                    errors.append((path, str(e)))
    return errors


if __name__ == "__main__":
    errors = validate_json_files()
    if errors:
        print("JSON validation errors:")
        for path, err in errors:
            print(f"  {path}: {err}")
        sys.exit(1)
    else:
        print("All JSON files are valid")
