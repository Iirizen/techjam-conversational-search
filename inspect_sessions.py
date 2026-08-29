import json
import sys

def load_sessions(path):
    sessions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions

def print_session(s):
    print("=" * 70)
    print("sample_id:       ", s.get("sample_id"))
    print("scenario_type:   ", s.get("scenario_type"))
    print("category_bucket: ", s.get("category_bucket"))
    print("difficulty:      ", s.get("difficulty_bucket"))
    print("target_asin:     ", s.get("ground_truth", {}).get("parent_asin"))
    print("user_profile:")
    print(json.dumps(s.get("user_profile", {}), indent=2))
    print()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/public_set.jsonl"
    scenario_filter = sys.argv[2] if len(sys.argv) > 2 else None
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    sessions = load_sessions(path)
    if scenario_filter:
        sessions = [s for s in sessions if s.get("scenario_type") == scenario_filter]

    print("Total matching sessions:", len(sessions))
    print()
    for s in sessions[:n]:
        print_session(s)
