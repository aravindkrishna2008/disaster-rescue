# 3D Asset Inventory

Downloaded for the disaster-rescue simulation and Three.js frontend.

## Kenney Factory Kit

- Path: `assets/3d/kenney_factory-kit`
- Source: https://www.kenney.nl/assets/factory-kit
- License: Creative Commons Zero (CC0)
- Useful files: `Models/GLB format/*.glb`
- Count: 143 GLB models
- Best uses: chemical plant, industrial pipes, warning signs, crates, conveyors, catwalks, machines, doors, factory props.

Original archive is kept at `assets/3d/kenney_factory-kit_3.0.zip`.

## Kenney Modular Dungeon Kit

- Path: `assets/3d/kenney_modular-dungeon-kit`
- Source: https://www.kenney.nl/assets/modular-dungeon-kit
- License: Creative Commons Zero (CC0)
- Useful files: `Models/GLB format/*.glb`
- Count: 39 GLB models
- Best uses: broken corridor pieces, walls, floors, stairs, gates, and modular enclosed-space layouts.

Original archive is kept at `assets/3d/kenney_modular-dungeon-kit_1.0.zip`.

## Low-Poly House Construction Site

- Path: `assets/3d/lowpoly-house-construction-site`
- Source: https://opengameart.org/content/3d-house-construction-site-lowpoly-cc0
- Author: Majadroid
- License: CC0; attribution not required but appreciated by the author.
- Useful files: `LowPoly-House-Construction-Site-By-Majadroid/fbx files/*.fbx`
- Count: 13 FBX models, plus Blender scene files
- Best uses: rubble containers, planks, barrels, trucks, construction fencing, roads, cranes, ramps, building shell.

Original archive is kept at `assets/3d/lowpoly-house-construction-site-by-majadroid.zip`.

## Integration Notes

- Prefer Kenney GLB files for the frontend because Three.js can load them directly with `GLTFLoader`.
- Prefer simple FBX/OBJ/converted meshes for MuJoCo visual assets, while keeping existing box/cylinder geoms for collision.
- Keep simulation collision geometry simple; use these models mainly as visual replacements for obstacles, hazards, and environment props.
