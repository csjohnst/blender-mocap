"""Dump all pose bone names from the active Rigify rig.

Run in Blender: blender --background --python scripts/dump_rigify_bones.py
Or paste into Blender's scripting tab.
"""
import bpy

# Create a metarig and generate
bpy.ops.object.armature_human_metarig_add()
metarig = bpy.context.active_object

# Generate the rig
bpy.ops.pose.rigify_generate()

# Find the generated rig (has rig_id)
rig = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE' and 'rig_id' in obj.data:
        rig = obj
        break

if rig is None:
    print("ERROR: No generated rig found")
else:
    print(f"\n=== Generated Rigify Rig: {rig.name} ===")
    print(f"Total bones: {len(rig.pose.bones)}")

    # Group by prefix
    groups = {}
    for pb in rig.pose.bones:
        name = pb.name
        if '-' in name:
            prefix = name.split('-')[0]
        else:
            prefix = 'CONTROL'
        groups.setdefault(prefix, []).append(name)

    for prefix in sorted(groups.keys()):
        bones = sorted(groups[prefix])
        print(f"\n--- {prefix} ({len(bones)} bones) ---")
        for b in bones:
            print(f"  {b}")

    # Specifically look for our target bones
    print("\n=== BONE NAME SEARCH ===")
    targets = ['upper_arm', 'forearm', 'hand', 'thigh', 'shin', 'foot',
               'spine', 'torso', 'head', 'neck']
    for target in targets:
        matches = [pb.name for pb in rig.pose.bones if target in pb.name.lower()]
        print(f"\n'{target}' matches:")
        for m in sorted(matches):
            print(f"  {m}")
