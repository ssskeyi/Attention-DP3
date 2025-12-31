#!/usr/bin/env python3
"""
Select geom ids from a body-geom map JSON by body-name keywords.

Usage:
  python tools/select_target_geomids_from_map.py <map.json> <keyword1,keyword2,...> <out.json>

Example:
  python tools/select_target_geomids_from_map.py body-geom-maps/door.json door,frame,latch targets/door_geom_ids.json

This script matches keywords as case-insensitive substrings of the `body_name`
or `geom_name` fields in the map file and writes a sorted unique list of geom ids.
"""
import json
import sys
import os

def load_map(path):
    with open(path, "r") as f:
        return json.load(f)

def select_by_keywords(map_list, keywords):
    """
    Select geom_ids by keywords.

    mode: 'body' => match keywords only against body_name
          'geom' => match keywords only against geom_name
          'both' => match either (legacy behavior)
    """
    kws = [k.strip().lower() for k in keywords if k.strip()]
    mode = None
    # allow passing mode as last element of keywords list like ['door','mode=both']
    filtered_kws = []
    for k in kws:
        if k.startswith("mode="):
            mode = k.split("=", 1)[1]
        else:
            filtered_kws.append(k)
    if mode is None:
        mode = "body"
    kws = filtered_kws

    selected = []
    for e in map_list:
        body = (e.get("body_name") or "").lower()
        geom = (e.get("geom_name") or "").lower()
        match = False
        if mode == "body":
            match = any(k in body for k in kws)
        elif mode == "geom":
            match = any(k in geom for k in kws)
        else:  # both
            match = any(k in body or k in geom for k in kws)
        if match:
            selected.append(int(e["geom_id"]))
    return sorted(set(selected))

def main():
    if len(sys.argv) < 3:
        print("Usage: select_target_geomids_from_map.py <map.json> <task_or_keyword1,keyword2,...> [out.json]")
        print("Example tasks: door, hammer, pen")
        sys.exit(1)
    map_path = sys.argv[1]
    key_arg = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Built-in keyword sets for common tasks (matching only against body_name)
    # add hand-related body keywords into each task's builtin set
    builtin = {
        "door": ["door", "frame", "latch", "forearm", "palm", "wrist", "ff", "mf", "rf", "th"],
        "hammer": ["object", "nail", "nail_board", "handle", "board", "forearm", "palm", "wrist", "ff", "mf", "rf", "th"],
        "pen": ["pen", "target", "object", "forearm", "palm", "wrist", "ff", "mf", "rf", "th"]
    }

    # Determine keywords: if key_arg matches a builtin task, use that set, otherwise parse comma list
    if key_arg.lower() in builtin:
        keywords = builtin[key_arg.lower()]
        # default output path if not provided
        if out_path is None:
            out_path = f"targets/{key_arg.lower()}_geom_ids.json"
    else:
        keywords = key_arg.split(",")
        if out_path is None:
            # try to make a reasonable default filename based on first keyword
            safe = key_arg.split(",")[0].strip().split()[0]
            out_path = f"targets/{safe}_geom_ids.json"

    if not os.path.exists(map_path):
        print(f"Map not found: {map_path}")
        sys.exit(1)

    m = load_map(map_path)
    sel = select_by_keywords(m, keywords)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sel, f, indent=2)
    print(f"Wrote {len(sel)} geom_ids to {out_path}")

if __name__ == "__main__":
    main()


