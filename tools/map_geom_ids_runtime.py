import sys, json

xml_path = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv)>2 else "geom_map.json"

# Prefer mujoco_py (provides geom_names) then fallback to mujoco with mj_name if available.
try:
    import mujoco_py
    from mujoco_py import load_model_from_path
    model = load_model_from_path(xml_path)
    mapping = []
    ngeom = model.ngeom
    for gid in range(ngeom):
        gname = model.geom_names[gid] if gid < len(model.geom_names) else None
        body_id = int(model.geom_bodyid[gid])
        body_name = model.body_names[body_id] if hasattr(model, "body_names") else str(body_id)
        # ensure geom_name is always present; if missing/empty, set placeholder
        gname_local = gname if gname and gname.strip() else f"geom_{gid}"
        # include geom position/size if available to help identify unlabeled geoms (e.g., door)
        try:
            geom_pos = list(map(float, model.geom_pos[gid]))
        except Exception:
            geom_pos = None
        try:
            geom_size = list(map(float, model.geom_size[gid]))
        except Exception:
            geom_size = None
        mapping.append({"geom_id": gid, "geom_name": gname_local, "body_id": body_id, "body_name": body_name,
                        "geom_pos": geom_pos, "geom_size": geom_size})
    with open(out_json, "w") as f:
        json.dump(mapping, f, indent=2)
    print("Wrote", out_json, "using 'mujoco_py' binding")
    sys.exit(0)
except Exception:
    pass

# Fallback to 'mujoco' binding (new bindings). Use mj_name if available.
try:
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml_path)
    mapping = []
    nges = getattr(model, "ngeom", None)
    if nges is None:
        # try length of geom_pos or other heuristic
        nges = int(getattr(model, "geom_pos", np.zeros((0,))).shape[0]) if 'np' in globals() else 0
    for gid in range(nges):
        try:
            name = mujoco.mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) if hasattr(mujoco, "mj_name") else None
        except Exception:
            name = None
        # ensure geom_name is always present
        name_local = name if name and name.strip() else f"geom_{gid}"
        try:
            body_id = int(model.geom_bodyid[gid])
            body_name = mujoco.mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) if hasattr(mujoco, "mj_name") else str(body_id)
        except Exception:
            body_id = -1
            body_name = str(body_id)
        # attempt to include pos/size arrays if present
        try:
            geom_pos = list(map(float, model.geom_pos[gid]))
        except Exception:
            geom_pos = None
        try:
            geom_size = list(map(float, model.geom_size[gid]))
        except Exception:
            geom_size = None
        mapping.append({"geom_id": gid, "geom_name": name_local, "body_id": body_id, "body_name": body_name,
                        "geom_pos": geom_pos, "geom_size": geom_size})
    with open(out_json, "w") as f:
        json.dump(mapping, f, indent=2)
    print("Wrote", out_json, "using 'mujoco' binding")
    sys.exit(0)
except Exception:
    pass

print("No supported MuJoCo Python binding found. Please install 'mujoco_py' or ensure 'mujoco' exposes mj_name; otherwise use the XML parser fallback.")