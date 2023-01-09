import torch
import safetensors
import safetensors.torch
import os
import copy

EMA_PREFIX = "model_ema."

METADATA = {'epoch': 0, 'global_step': 0, 'pytorch-lightning_version': '1.6.0'}

IDENTIFICATION = {
    "VAE": {
        "SD-v1": 0,
        "SD-v2": 869,
        "NAI": 2982,
        "WD-VAE-v1": 155,
        "WD-VAE-v2": 41
    },
    "CLIP-v1": {
        "SD-v1": 0,
    },
    "CLIP-v2": {
        "SD-v2": 1141,
        "WD-v1-4": 2543
    }
}

COMPONENTS = {
    "UNET-v1-SD": {
        "keys": {},
        "source": "UNET-v1-SD.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v1-EMA": {
        "keys": {},
        "source": "UNET-v1-EMA.txt",
        "prefix": "model_ema.diffusion_model"
    },
    "UNET-v1-Inpainting": {
        "keys": {},
        "source": "UNET-v1-Inpainting.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v2-SD": {
        "keys": {},
        "source": "UNET-v2-SD.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v2-Depth": {
        "keys": {},
        "source": "UNET-v2-Depth.txt",
        "prefix": "model.diffusion_model."
    },
    "VAE-v1-SD": {
        "keys": {},
        "source": "VAE-v1-SD.txt",
        "prefix": "first_stage_model."
    },
    "CLIP-v1-SD": {
        "keys": {},
        "source": "CLIP-v1-SD.txt",
        "prefix": "cond_stage_model.transformer.text_model."
    },
    "CLIP-v1-NAI": {
        "keys": {},
        "source": "CLIP-v1-SD.txt",
        "prefix": "cond_stage_model.transformer."
    },
    "CLIP-v2-SD": {
        "keys": {},
        "source": "CLIP-v2-SD.txt",
        "prefix": "cond_stage_model.model."
    },
    "CLIP-v2-WD": {
        "keys": {},
        "source": "CLIP-v2-WD.txt",
        "prefix": "cond_stage_model.model."
    },
    "Depth-v2-SD": {
        "keys": {},
        "source": "Depth-v2-SD.txt",
        "prefix": "depth_model.model."
    }
}

COMPONENT_CLASS = {
    "UNET-v1-SD": "UNET-v1",
    "UNET-v1-EMA": "EMA-UNET-v1",
    "UNET-v1-Inpainting": "UNET-v1",
    "UNET-v2-SD": "UNET-v2",
    "UNET-v2-Depth": "UNET-v2-Depth",
    "VAE-v1-SD": "VAE-v1",
    "CLIP-v1-SD": "CLIP-v1",
    "CLIP-v1-NAI": "CLIP-v1",
    "CLIP-v2-SD": "CLIP-v2",
    "CLIP-v2-WD": "CLIP-v2",
    "Depth-v2-SD": "Depth-v2"
}

OPTIONAL = [
    ("alphas_cumprod", (1000,)),
    ("alphas_cumprod_prev", (1000,)),
    ("betas", (1000,)),
    ("log_one_minus_alphas_cumprod", (1000,)),
    ("model_ema.decay", ()),
    ("model_ema.num_updates", ()),
    ("posterior_log_variance_clipped", (1000,)),
    ("posterior_mean_coef1", (1000,)),
    ("posterior_mean_coef2", (1000,)),
    ("posterior_variance", (1000,)),
    ("sqrt_alphas_cumprod", (1000,)),
    ("sqrt_one_minus_alphas_cumprod", (1000,)),
    ("sqrt_recip_alphas_cumprod", (1000,)),
    ("sqrt_recipm1_alphas_cumprod", (1000,))
]


ARCHITECTURES = {
    "UNET-v1": {
        "classes": ["UNET-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "UNET-v2": {
        "classes": ["UNET-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "UNET-v2-Depth": {
        "classes": ["UNET-v2-Depth"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "VAE-v1": {
        "classes": ["VAE-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "CLIP-v1": {
        "classes": ["CLIP-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "CLIP-v2": {
        "classes": ["CLIP-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "Depth-v2": {
        "classes": ["Depth-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "SD-v1": {
        "classes": ["UNET-v1", "VAE-v1", "CLIP-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v2": {
        "classes": ["UNET-v2", "VAE-v1", "CLIP-v2"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v2-Depth": {
        "classes": ["UNET-v2-Depth", "VAE-v1", "CLIP-v2", "Depth-v2"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "EMA-v1": {
        "classes": ["EMA-UNET-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    # standalone component architectures, for detecting broken models
    "UNET-v1-BROKEN": {
        "classes": ["UNET-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "UNET-v2-BROKEN": {
        "classes": ["UNET-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "UNET-v2-Depth-BROKEN": {
        "classes": ["UNET-v2-Depth"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "VAE-v1-BROKEN": {
        "classes": ["VAE-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "CLIP-v1-BROKEN": {
        "classes": ["CLIP-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "CLIP-v2-BROKEN": {
        "classes": ["CLIP-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "Depth-v2-BROKEN": {
        "classes": ["Depth-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    }
}

def tensor_size(t):
    if type(t) == torch.Tensor:
        return t.nelement() * t.element_size()
    return 0

def tensor_shape(data):
    if hasattr(data, 'shape'):
        return tuple(data.shape)
    return tuple()

def load_components(path):
    for c in COMPONENTS:
        file = os.path.join(path, COMPONENTS[c]["source"])
        if not os.path.exists(file):
            print(f"CANNOT FIND {c} KEYS")
        with open(file, 'r') as f:
            COMPONENTS[c]["keys"] = set()
            for l in f:
                l = l.rstrip().split(" ")
                k, z = l[0], l[1]
                z = z[1:-1].split(",")
                if not z[0]:
                    z = tuple()
                else:
                    z = tuple(int(i) for i in z)
                COMPONENTS[c]["keys"].add((k,z))

def get_prefixed_keys(component):
    prefix = COMPONENTS[component]["prefix"]
    allowed = COMPONENTS[component]["keys"]
    return set([(prefix + k, z) for k, z in allowed])

def get_keys_size(model, keys):
    z = 0
    for k in keys:
        if k in model:
            z += tensor_size(model[k])
    return z

class FakeTensor():
    def __init__(self, shape):
        self.shape = shape

def build_fake_model(model):
    fake_model = {}
    for k in model:
        fake_model[k] = FakeTensor(tensor_shape(model[k]))
    return fake_model

def inspect_model(model, all=False):
    # find all arch's and components in the model
    # also reasons for failing to find them

    keys = set([(k, tensor_shape(model[k])) for k in model])

    rejected = {}

    components = [] # comp -> prefixed
    classes = {} # class -> [comp]
    for comp in COMPONENTS:
        required_keys_unprefixed = COMPONENTS[comp]["keys"]
        required_keys_prefixed = get_prefixed_keys(comp)
        missing_unprefixed = required_keys_unprefixed.difference(keys)
        missing_prefixed = required_keys_prefixed.difference(keys)

        if not missing_unprefixed:
            components += [(comp, False)]
        if not missing_prefixed:
            components += [(comp, True)]

        if missing_prefixed and missing_unprefixed:
            if missing_prefixed != required_keys_prefixed:
                rejected[comp] = rejected.get(comp, []) + [{"reason": f"Missing required keys ({len(missing_prefixed)} of {len(required_keys_prefixed)})", "data": list(missing_prefixed)}]
            
            if missing_unprefixed != required_keys_unprefixed:
                rejected[comp] = rejected.get(comp, []) + [{"reason": f"Missing required keys ({len(missing_unprefixed)} of {len(required_keys_unprefixed)})", "data": list(missing_unprefixed)}]
        else:
            clss = COMPONENT_CLASS[comp]
            classes[clss] = [comp] + classes.get(clss, [])
    
    found = {} # arch -> {class -> [comp]}
    for arch in ARCHITECTURES:
        needs_prefix = ARCHITECTURES[arch]["prefixed"]
        required_classes = set(ARCHITECTURES[arch]["classes"])
        required_keys = set(ARCHITECTURES[arch]["required"])

        if not required_keys.issubset(keys):
            missing = required_keys.difference(keys)
            if missing != required_keys:
                rejected[arch] = rejected.get(arch, []) + [{"reason": f"Missing required keys ({len(missing)} of {len(required_keys)})", "data": list(missing)}]
            continue

        found_classes = {}
        for clss in required_classes:
            if clss in classes:
                for comp in classes[clss]:
                    
                    if (comp, needs_prefix) in components:# or ((comp, not needs_prefix) in components and not needs_prefix):
                        found_classes[clss] = found_classes.get(clss, [])
                        found_classes[clss] += [comp]
                    #else:
                    #    rejected[arch] = rejected.get(arch, []) + [{"reason": "Class has incorrect prefix", "data": [clss]}]

        found_class_names = set(found_classes.keys())
        if not required_classes.issubset(found_class_names):
            if found_class_names:
                missing = list(required_classes.difference(found_class_names))
                rejected[arch] = rejected.get(arch, []) + [{"reason": "Missing required classes", "data": missing}]
            continue

        found[arch] = found_classes

    # if we found a real architecture then dont show the broken ones
    if any([not a.endswith("-BROKEN") for a in found]):
        for a in list(found.keys()):
            if a.endswith("-BROKEN"):
                del found[a]

    if all:
        return found, rejected
    else:
        return resolve_arch(found)

def resolve_class(components):
    components = list(components)

    if not components or len(components) == 1:
        return components

    # prefer SD components vs busted ass components
    sd_components = [c for c in components if "SD" in c]
    if len(sd_components) == 1:
        return [sd_components[0]]

    # otherwise component with the most keys is probably the best
    components = sorted(components, key=lambda c: len(COMPONENTS[c]["keys"]), reverse=True)

    return [components[0]]

def resolve_arch(arch):
    arch = copy.deepcopy(arch)
    # resolve potentially many overlapping arch's to a single one

    if not arch:
        return {}

    # select arch with most keys
    arch_sizes = {}
    for a in arch:
        arch_sizes[a] = len(ARCHITECTURES[a]["required"])
        for clss in arch[a]:
            arch[a][clss] = resolve_class(arch[a][clss])
            if arch[a][clss]:
                arch_sizes[a] += len(COMPONENTS[arch[a][clss][0]]["keys"])
    choosen = max(arch_sizes, key=arch_sizes.get)
    return {choosen: arch[choosen]}

def find_components(arch, component_class):
    components = set()
    for a in arch:
        if component_class in arch[a]:
            components.update(arch[a][component_class])
    return components

def contains_component(model, component, prefixed = None):
    model_keys = set([(k, tensor_shape(model[k])) for k in model])

    allowed = False
    if prefixed == None: #prefixed or unprefixed
        allowed = get_prefixed_keys(component).issubset(model_keys)
        allowed = allowed or COMPONENTS[component]["keys"].issubset(model_keys)
    elif prefixed == True:
        allowed = get_prefixed_keys(component).issubset(model_keys)
    elif prefixed == False:
        allowed = COMPONENTS[component]["keys"].issubset(model_keys)

    return allowed

def get_allowed_keys(arch, allowed_classes=None):
    # get all allowed keys
    allowed = set()
    for a in arch:
        if allowed_classes == None:
            allowed.update(ARCHITECTURES[a]["required"])
            allowed.update(ARCHITECTURES[a]["optional"])
        prefixed = ARCHITECTURES[a]["prefixed"]
        for clss in arch[a]:
            if allowed_classes == None or clss in allowed_classes:
                for comp in arch[a][clss]:
                    comp_keys = COMPONENTS[comp]["keys"]
                    if prefixed:
                        comp_keys = get_prefixed_keys(comp)
                    allowed.update(comp_keys)
    return allowed

def fix_model(model, fix_clip=False):
    # fix NAI nonsense
    nai_keys = {
        'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
        'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
        'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.'
    }
    renamed = False
    for k in list(model.keys()):
        for r in nai_keys:
            if k.startswith(r):
                renamed = True
                kk = k.replace(r, nai_keys[r])
                model[kk] = model[k]
                del model[k]
                break
    
    # fix merging nonsense
    i = "cond_stage_model.transformer.text_model.embeddings.position_ids"
    if i in model:
        if fix_clip:
            # actually fix the ids
            model[i] = torch.Tensor([list(range(77))]).to(torch.int64)
        else:
            # ensure fp16 looks the same as fp32
            model[i] = model[i].to(torch.int64)

    return renamed

def fix_ema(model):
    # turns UNET-v1-EMA into UNET-v1-SD
    # but only when in component form (unprefixed)

    # example keys
    # EMA = model_ema.diffusion_modeloutput_blocks91transformer_blocks0norm3weight
    # SD  = model.diffusion_model.output_blocks9.1.transformer_blocks.0.norm3.weight

    normal = COMPONENTS["UNET-v1-SD"]["keys"]
    for k, _ in normal:
        kk = k.replace(".", "")
        if kk in model:
            model[k] = model[kk]
            del model[kk]
def compute_metric(model, arch=None):
    def tensor_metric(t):
        t = t.to(torch.float16).to(torch.float32)
        return torch.sum(torch.sigmoid(t)-0.5)

    if arch == None:
        arch = inspect_model(model)

    unet_keys = get_allowed_keys(arch, ["UNET-v1", "UNET-v2", "UNET-v2-Depth"])
    vae_keys = get_allowed_keys(arch, ["VAE-v1"])
    clip_keys = get_allowed_keys(arch, ["CLIP-v1", "CLIP-v2"])

    unet, vae, clip = 0, 0, 0

    is_clip_v1 = "CLIP-v1" in next(iter(arch.values()))

    for k in model:
        kk = (k, tensor_shape(model[k]))

        if kk in unet_keys:
            unet += tensor_metric(model[k])

        if kk in vae_keys:
            if "encoder." in k or "decoder." in k:
                vae += tensor_metric(model[k])

        if kk in clip_keys:
            if "mlp." in k and not ".23." in k:
                clip += tensor_metric(model[k])

    b_unet, b_vae, b_clip = -6131.5400, 17870.7051, -2097.8596 if is_clip_v1 else -8757.5630
    k_unet, k_vae, k_clip = 10000, 10000, 1000000 if is_clip_v1 else 10000

    r = 10000

    n_unet = int(abs(unet/b_unet - 1) * k_unet)
    n_vae = int(abs(vae/b_vae - 1) * k_vae)
    n_clip = int(abs(clip/b_clip - 1) * k_clip)

    while n_unet >= r:
        n_unet -= r//2
    
    while n_vae >= r:
        n_vae -= r//2

    while n_clip >= r:
        n_clip -= r//2

    s_unet = f"{n_unet:04}" if unet != 0 else "----"
    s_vae = f"{n_vae:04}" if vae != 0 else "----"
    s_clip = f"{n_clip:04}" if clip != 0 else "----"

    n_unet = None if unet == 0 else n_unet
    n_vae = None if vae == 0 else n_vae
    n_clip = None if clip == 0 else n_clip
    
    return s_unet+"/"+s_vae+"/"+s_clip, (n_unet, n_vae, n_clip)

def load(file):
    model = {}
    metadata = {}

    if file.endswith(".safetensors") or file.endswith(".st"):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                model[key] = f.get_tensor(key)
    else:
        model = torch.load(file, map_location="cpu")
        if not model:
            return {}, {}
        if 'state_dict' in model:
            for k in model:
                if k != 'state_dict':
                    metadata[k] = model[k]
            model = model['state_dict']

    return model, metadata

def save(model, metadata, file):
    if file.endswith(".safetensors"):
        safetensors.torch.save_file(model, file)
        return
    else:
        out = metadata
        out['state_dict'] = model
        torch.save(out, file)

def prune_model(model, arch, keep_ema, dont_half):
    allowed = get_allowed_keys(arch)
    for k in list(model.keys()):
        kk = (k, tensor_shape(model[k]))
        keep = False
        if kk in allowed:
            keep = True
        if k.startswith(EMA_PREFIX) and keep_ema:
            keep = True
        if not keep:
            del model[k]
            continue
        if not dont_half and type(model[k]) == torch.Tensor and model[k].dtype == torch.float32:
            model[k] = model[k].half()

def extract_component(model, component, prefixed=None):
    prefix = COMPONENTS[component]["prefix"]
    allowed = set()
    if prefixed != True:
        allowed = allowed.union(COMPONENTS[component]["keys"])
    if prefixed != False:
        allowed = allowed.union(get_prefixed_keys(component))

    for k in list(model.keys()):
        z = tensor_shape(model[k])
        if (k, z) in allowed:
            if k.startswith(prefix):
                kk = k.replace(prefix,"")
                model[kk] = model[k]
                del model[k]
        else:
            del model[k]

def replace_component(target, target_arch, source, source_component):
    if not COMPONENT_CLASS[source_component] in ARCHITECTURES[target_arch]["classes"]:
        raise ValueError(f"{target_arch} cannot contain {source_component}!")

    # get component for class
    prefix = COMPONENTS[source_component]["prefix"]
    component_keys = COMPONENTS[source_component]["keys"]

    # find out if we should prefix the component
    is_prefixed = ARCHITECTURES[target_arch]["prefixed"]

    for k in list(source.keys()):
        src_z = tensor_shape(source[k])
        src_k = k[len(prefix):] if k.startswith(prefix) else k
        dst_k = prefix + k if is_prefixed else k
        if (src_k, src_z) in component_keys:
            target[dst_k] = source[k]

def delete_class(model, model_arch, component_class):
    keys = set([(k, tensor_shape(model[k])) for k in model])
    prefixed = ARCHITECTURES[model_arch]["prefixed"]

    for name, component in COMPONENTS.items():
        if COMPONENT_CLASS[name] != component_class:
            continue
        component_keys = component["keys"] if not prefixed else get_prefixed_keys(name)
        for k in component_keys:
            if k in keys:
                del model[k[0]]
                keys.remove(k)

def log(model, file):
    keys = []
    for k in model:
        size = str(list(model[k].shape))
        keys += [f"{k},{size}"]
    keys.sort()
    out = "\n".join(keys)
    with open(file, "w") as f:
        f.write(out)

if __name__ == '__main__':
    import glob 
    r = "/run/media/pul/ssd/stable-diffusion-webui/models/Stable-diffusion/**/*fumo-800.ckpt"

    print(glob.glob(r, recursive=True))

    exit()

    a, _ = load(r)

    print(list(a.keys()))

    load_components("components")

    extract_component(a, "UNET-v1-EMA")

    print(list(a.keys()))