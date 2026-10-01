# model3d: build 3D parts from primitives

`build_model.py`, `model3d.py` and `inspect3d.py` were copied unchanged from the `pro-product-video` skill
(`product-video` / `pro-product-video`, scripts folder). Together they do three things:

- turn a JSON spec of primitives into a GLB with named parts, PBR materials and explode offsets;
- render that GLB in software (pure numpy + OpenCV);
- inspect it: make 6-view sheets and pick 3D points.

Used by `scripts/deployment/build_component_models.py`. That script builds the 3D component models of the Android
app from the OBD2 knowledge base (`configs/diagnosis_knowledge.yaml`). The same GLBs can feed the skill's
3D video mode, e.g. `"model": {"file": "battery.glb"}` in a storyboard.
