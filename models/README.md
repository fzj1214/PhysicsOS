# PhysicsOS Model Checkpoints

Put real trained surrogate/neural-operator weights under this directory.

Recommended layout:

```text
models/
  grid_fno_local/
    model.ts
    metadata.json
  geometry_informed_no_local/
    model.ts
    metadata.json
  mesh_graph_operator_local/
    model.ts
    metadata.json
```

Supported first:

```text
TorchScript checkpoint: *.ts / *.pt with format="torchscript"
PyTorch state_dict: *.pt / *.pth with format="torch_state_dict" requires a model factory plugin
Safetensors: *.safetensors with format="safetensors" requires a model factory plugin
```

Do not commit large checkpoints unless this repository is configured for Git LFS.

