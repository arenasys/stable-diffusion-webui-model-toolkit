import os
import sys
import gradio as gr
from modules import shared, script_callbacks
import torch
import glob

from toolkit import *

ROOT_PATH = "models"
MODEL_PATH = os.path.join(ROOT_PATH, "Stable-diffusion")
COMPONENT_PATH = os.path.join(ROOT_PATH, "Components")
VAE_PATH = os.path.join(ROOT_PATH, "VAE")

MODEL_EXT = [".ckpt", ".pt", ".safetensors"]
COMPONENT_EXT = {
    "UNET-v1": ".unet.pt", 
    "EMA-UNET-v1": ".unet.pt", 
    "UNET-v2": ".unet-v2.pt", 
    "UNET-v2-Depth": ".unet-v2-d.pt", 
    "VAE-v1": ".vae.pt", 
    "CLIP-v1": ".clip.pt", 
    "CLIP-v2": ".clip-v2.pt", 
    "Depth-v2": ".depth.pt"
}

os.makedirs(COMPONENT_PATH, exist_ok=True)

class ToolkitModel():
    def __init__(self):
        self.filename = ""
        self.model = {}
        self.metadata = {}

        self.partial = False

        self.a_found = {}
        self.a_rejected = {}
        self.a_resolved = {}
        self.a_potential = []
        self.a_type = ""
        self.a_classes = {}
        self.a_components = []

        self.m_str = "----/----/----"
        self.m_unet = None
        self.m_vae = None
        self.m_clip = None
    
        self.z_total = 0
        self.z_full = 0
        self.z_junk = 0
        self.z_ema = 0

        self.k_junk = []
        self.k_ema = []

def do_analysis(model):
    tm = ToolkitModel()
    tm.model = model

    tm.a_found, tm.a_rejected = inspect_model(model, all=True)
    tm.a_resolved = resolve_arch(tm.a_found)
    
    if not tm.a_resolved:
        tm.m_str = "----/----/----"
        return tm

    tm.a_potential = list(tm.a_found.keys())
    tm.a_type = next(iter(tm.a_resolved))
    tm.a_classes = tm.a_resolved[tm.a_type]
    tm.a_components = [tm.a_classes[c][0] for c in tm.a_classes]
    
    tm.m_str, m = compute_metric(model, tm.a_resolved)
    tm.m_unet, tm.m_vae, tm.m_clip = m

    allowed = get_allowed_keys(tm.a_resolved)

    for k in model.keys():
        kk = (k, tensor_shape(model[k]))
        z = tensor_size(model[k])
        tm.z_total += z

        if kk in allowed:
            if z and model[k].dtype == torch.float32:
                tm.z_full += z
        else:
            if k.startswith(EMA_PREFIX):
                tm.z_ema += z
                tm.k_ema += [k]
            else:
                tm.z_junk += z
                tm.k_junk += [k]
    
    return tm

def get_size(bytes):
    KB = 1024
    MB = KB * 1024
    GB = MB * 1024

    if bytes < KB:
        return f"{bytes} Bytes"
    if bytes < MB:
        return f"{bytes/KB:.2f} KB"
    if bytes < 1024*1024*1024:
        return f"{bytes/MB:.2f} MB"
    return f"{bytes/GB:.2f} GB"

def do_basic_report(details: ToolkitModel, dont_half, keep_ema):
    d = details

    report = f"### Report ({d.m_str})\n-----\n"

    if not d.a_found:
        report += "Model type could not be identified.\n\n"
        return report

    out = [f"Model is **{get_size(d.z_total)}**."]

    if len(d.a_potential) > 1:
        out += [f"Multiple model types identified: **{', '.join(d.a_potential)}**."]
        out += [f"Model type **{d.a_type}** will be used."]
    else:
        out += [f"Model type identified as **{d.a_type}**."]

    if d.a_components:
        out += [f"Model components are: **{', '.join(d.a_components)}**."]
    else:
        out += [f"**Model has no components**."]
    
    report += " ".join(out) + "\n\n"
    out = []

    k_junk = d.k_junk
    z_junk = d.z_junk
    if not keep_ema:
        k_junk += d.k_ema
        z_junk += d.z_ema
    
    if k_junk:
        if z_junk > 16777216:
            out += [f"**Contains {get_size(z_junk)} of junk data!**"]
        else:
            out += [f"**Contains {len(k_junk)} junk keys!**"]
    else:
        out += [f"Contains no junk data."]

    if keep_ema:
        if d.k_ema:
            if d.z_ema > 16777216:
                out += [f"**Contains {get_size(d.z_ema)} of EMA data!**"]
            else:
                out += [f"**Contains {len(d.k_ema)} EMA keys!**"]
        else:
            out += [f"Contains no EMA data."]
    
    if d.z_full > 0:
        out += [f"Wastes **{get_size(d.z_full//2)}** on precision."]

    if "CLIP-v1-NAI" in d.a_components:
        out += ["**CLIP is mislablled.**"]
    if "CLIP-v2-WD" in d.a_components:
        out += ["**CLIP is missing its final layer.**"]

    report += " ".join(out) + "\n\n"
    out = []

    if d.m_vae != None:
        for k, i in IDENTIFICATION["VAE"].items():
            if abs(d.m_vae-i) <= 1:
                out += [f"Uses the **{k}** VAE."]

    if d.m_clip != None:
        for k, i in IDENTIFICATION["CLIP-v1"].items():
            if abs(d.m_clip-i) <= 1:
                out += [f"Uses the **{k}** CLIP."]

        for k, i in IDENTIFICATION["CLIP-v2"].items():
            if abs(d.m_clip-i) <= 1:
                out += [f"Uses the **{k}** CLIP."]

    report += " ".join(out) + "\n\n"

    removed = d.z_junk
    if not keep_ema:
        removed += d.z_ema
    if not dont_half:
        removed += d.z_full//2

    if (d.z_full and not dont_half) or d.k_ema or d.k_junk:
        report += f"Model can be pruned to **{get_size(d.z_total-removed)}**."
    else:
        report += f"**Model is clean, nothing to be done.**"

    return report


def do_adv_report(details: ToolkitModel, abbreviate=True):
    d = details
    
    model_keys = set((k, tensor_shape(d.model[k])) for k in d.model.keys())
    allowed_keys = get_allowed_keys(d.a_resolved)
    known_keys = get_allowed_keys(d.a_found)

    unknown_keys = model_keys.difference(known_keys)
    useless_keys = known_keys.difference(allowed_keys)

    model_size, useless_size, unknown_size = 0, 0, 0

    for k in d.model:
        kk = (k, tensor_shape(d.model[k]))
        z = tensor_size(d.model[k])
        model_size += z
        if kk in useless_keys:
            useless_size += z
        if kk in unknown_keys:
            unknown_size += z

    report = f"### Report ({d.m_str})\n-----\n"
    report += f"#### Statistics\n"
    line = [f"Total keys: **{len(model_keys)} ({get_size(model_size)})**"]
    if useless_keys:
        line += [f"Useless keys: **{len(useless_keys)} ({get_size(useless_size)})**"]
    if unknown_keys:
        line += [f"Unknown keys: **{len(unknown_keys)} ({get_size(unknown_size)})**"]

    report += ", ".join(line) + ".\n"
    report += "#### Architecture\n"

    if d.a_found:
        archs = [d.a_type] + [a for a in d.a_potential if not a == d.a_type]
        for arch in archs:
            report += f"- **{arch}**\n"

            data = d.a_found[arch]
            if arch == d.a_type and abbreviate:
                data = d.a_resolved[arch]
                
            for clss in data:
                report += f"  - **{clss}**\n"
                for comp in data[clss]:
                    report += f"    - {comp}\n"
                if not data[clss]:
                    report += f"    - NONE\n"
            if len(archs) > 1 and arch == d.a_type:
                report += f"#### Additional\n"
    else:
        report += "- *NONE*\n"

    if d.a_rejected:
        report += "#### Rejected\n"
        for arch, rejs in d.a_rejected.items():
            for rej in rejs:
                report += f"- **{arch}**: {rej['reason']}\n"
                data = list(rej["data"])
                will_abbreviate = len(rej["data"]) > 5 and abbreviate
                if will_abbreviate:
                    data = data[:5]
                for k in data:
                    if type(k) == tuple:
                        report += f"  - {k[0]} {k[1]}\n"
                    else:
                        report += f"  - {k}\n"
                if will_abbreviate:
                    report += f"  - ...\n"


    if unknown_keys:
        report += "#### Unknown\n"

        data = sorted(unknown_keys)
        will_abbreviate = len(data) > 5 and abbreviate
        if will_abbreviate:
            data = data[5:]
        for k, z in data:
            report += f" - {k} {z}\n"
        if will_abbreviate:
            report += f" - ...\n"

    return report

source_list = []
file_list = []
loaded = None

def get_models(dir):
    ext = ["**" + os.sep + "*" + e for e in MODEL_EXT]
    files = []
    for e in ext:
        files += glob.glob(dir + os.sep + e, recursive=True)
    return files

def get_lists():
    global source_list, file_list
    file_list = get_models(COMPONENT_PATH)
    file_list += get_models(MODEL_PATH)
    file_list += get_models(VAE_PATH)

    unique_list = []
    dup_list = []
    for a in file_list:
        collision = False
        for b in file_list:
            if a != b and a[a.rfind(os.sep):] == b[b.rfind(os.sep):]:
                collision = True
                break
        if collision:
            dup_list += [a]
        else:
            unique_list += [a]
    unique_list = sorted([f[f.rfind(os.sep)+1:] for f in unique_list])

    dup_list = sorted([f[len(ROOT_PATH)+1:] for f in dup_list])

    file_list = unique_list + dup_list

    source_list = [] + file_list + [""]
    for a in ARCHITECTURES:
        if a.startswith("SD"):
            source_list += ["NEW " + a]

def find_source(source):
    if not source:
        return None
    if os.sep in source:
        s = os.path.join(ROOT_PATH, source)
        if os.path.exists(s):
            return s
        else:
            return None
    else:
        paths = [MODEL_PATH, VAE_PATH, COMPONENT_PATH]
        for p in paths:
            s = glob.glob(os.path.join(p, "**", "*" + source), recursive=True)
            if s:
                return s[0]
        return None

def get_name(tm: ToolkitModel, arch):
    name = "model"
    if tm.filename:
        name = os.path.basename(tm.filename).split(".")[0]
    
    if tm.m_str:
        if "SD" in arch:
            name += "-" + tm.m_str.replace("/","-")
        else:
            m = tm.m_str.split("/")
            if "UNET" in arch:
                name += "-" + m[0]
            elif "VAE" in arch:
                name += "-" + m[1]
            elif "CLIP" in arch:
                name += "-" + m[2]

    if arch in COMPONENT_EXT:
        name += COMPONENT_EXT[arch]
    else:
        name += ".safetensors"
    return name
    
def do_load(source, precision):
    global loaded
    
    basic_report, adv_report, save_name, error = "", "", "", ""

    dont_half = "FP32" in precision
    keep_ema = False

    loaded = None

    if source.startswith("NEW "):
        loaded = ToolkitModel()
        model_type = source[4:]
        for clss in ARCHITECTURES[model_type]["classes"]:
            loaded.a_classes[clss] = []
        loaded.a_type = model_type
        loaded.a_resolved = {model_type: loaded.a_classes}
        loaded.a_found = loaded.a_resolved
        loaded.a_potential = loaded.a_resolved

        loaded.partial = True
    else:
        filename = find_source(source)
        if not filename:
            error = f"Cannot find {source}!"
        else:
            model, _ = load(filename)
            fix_model(model, fix_clip=shared.opts.model_toolkit_fix_clip)
            loaded = do_analysis(model)
            loaded.filename = filename
    if loaded:
        basic_report = do_basic_report(loaded, dont_half, keep_ema)
        adv_report = do_adv_report(loaded)
        save_name = get_name(loaded, loaded.a_type)

    if error:
        error = f"### ERROR: {error}\n----"


    reports = [gr.update(value=basic_report), gr.update(value=adv_report)]
    sources = [gr.update(), gr.update()]
    drops = [gr.update(choices=[], value="") for _ in range(3)]
    rows = [gr.update(visible=not loaded), gr.update(visible=not not loaded)]
    names = [gr.update(value=save_name), gr.update()]
    error = [gr.update(value=error), gr.update(visible=not not error)]

    if loaded and loaded.a_found:
        drop_arch_list = sorted(loaded.a_potential)
        drops = [gr.update(choices=drop_arch_list, value=loaded.a_type)]

        drop_class_list = sorted(loaded.a_found[loaded.a_type].keys())
        drops += [gr.update(choices=drop_class_list, value=drop_class_list[0])]

        drop_comp_list = ["auto"] + sorted(loaded.a_found[loaded.a_type][drop_class_list[0]])
        drops += [gr.update(choices=drop_comp_list, value="auto")]

    updates = reports + sources + drops + rows + names + error
    return updates

def do_select(drop_arch, drop_class, drop_comp):
    global loaded
    if not loaded:
        return [gr.update(choices=[], value="") for _ in range(3)] + [gr.update(value="")]

    arch_list = sorted(loaded.a_potential)
    if not drop_arch in arch_list:
        drop_arch = loaded.a_type

    arch = loaded.a_found[drop_arch]

    class_list = sorted(arch.keys())
    if not drop_class in class_list:
        drop_class = class_list[0]

    comps = arch[drop_class]
    
    comp_list = ["auto"] + sorted(comps)
    if not drop_comp in comp_list:
        drop_comp = comp_list[0]

    export_name = get_name(loaded, drop_class)

    updates = [
        gr.update(choices=arch_list, value=drop_arch),
        gr.update(choices=class_list, value=drop_class),
        gr.update(choices=comp_list, value=drop_comp),
        gr.update(value=export_name)
    ]

    return updates

def do_clear():
    global loaded
    loaded = None
    
    reports = [gr.update(value=""), gr.update(value="")]
    sources = [gr.update(), gr.update()]
    drops = [gr.update(choices=[], value="") for _ in range(3)]
    rows = [gr.update(visible=True), gr.update(visible=False)]
    names = [gr.update(value=""), gr.update()]
    error = [gr.update(value=""), gr.update(visible=False)]

    updates = reports + sources + drops + rows + names + error
    return updates

def do_refresh():
    get_lists()
    return [gr.update(choices=source_list, value=source_list[0]), gr.update(choices=file_list, value=source_list[0])]

def do_report(precision):
    dont_half = "FP32" in precision
    keep_ema = False

    basic_report = do_basic_report(loaded, dont_half, keep_ema)
    adv_report = do_adv_report(loaded)
    save_name = get_name(loaded, loaded.a_type)

    values = [basic_report, adv_report, save_name]

    return [gr.update(value=v) for v in values]

def do_save(save_name, precision):
    dont_half = "FP32" in precision
    keep_ema = False

    folder = COMPONENT_PATH
    if "SD" in loaded.a_type:
        folder = MODEL_PATH
    elif "VAE" in loaded.a_type:
        folder = VAE_PATH

    filename = os.path.join(folder, save_name)
    model = copy.deepcopy(loaded.model)

    prune_model(model, loaded.a_resolved, keep_ema, dont_half)

    error = ""

    if model:
        save(model, METADATA, filename)
    else:
        error = f"### ERROR: Model is empty!\n----"
    del model

    reports = [gr.update(), gr.update()]
    sources = [gr.update(), gr.update()]
    drops = [gr.update() for _ in range(3)]
    rows = [gr.update(), gr.update()]
    names = [gr.update(), gr.update()]
    error = [gr.update(value=error), gr.update(visible=not not error)]

    updates = reports + sources + drops + rows + names + error
    return updates

def do_export(drop_arch, drop_class, drop_comp, export_name):
    error = ""

    if not loaded or not loaded.model:
        error = f"### ERROR: Model is empty!\n----"
    else:
        comp = drop_comp

        if comp == "auto":
            comp = resolve_class(loaded.a_found[drop_arch][drop_class])
            if comp:
                comp = comp[0]

        if comp:
            if not contains_component(loaded.model, comp):
                error = f"### ERROR: Model doesnt contain a {comp}!\n----"

            model = build_fake_model(loaded.model)
            prefixed = ARCHITECTURES[drop_arch]["prefixed"]
            prefix = COMPONENTS[comp]["prefix"]

            extract_component(model, comp, prefixed)
            
            for k in model:
                kk = prefix + k if prefixed else k
                model[k] = loaded.model[kk]

            if "EMA" in comp:
                fix_ema(model)

            folder = COMPONENT_PATH
            if "VAE" in comp:
                folder = VAE_PATH

            filename = os.path.join(folder, export_name)
            save(model, {}, filename)
        else:
            error = f"### ERROR: Model doesnt contain a {drop_class}!\n----"
            pass
    
    updates = [gr.update() for _ in range(4)] + [gr.update(value=error), gr.update(visible=not not error)]
    return updates

def do_import(drop_arch, drop_class, drop_comp, import_drop, precision):
    global loaded
    error = ""

    if not loaded or not import_drop:
        error = "### ERROR: No model is loaded!\n----"
    
    if not error:
        filename = find_source(import_drop)
        model, _ = load(filename)
        fix_model(model, fix_clip=shared.opts.model_toolkit_fix_clip)
        found, _ = inspect_model(model, all=True)
        if not found or not model:
            error = "### ERROR: Imported model could not be identified!\n----"
        
    choosen = ""
    if not error:
        # find all the components in the class
        possible = find_components(found, drop_class)
        if not possible:
            error = f"### ERROR: Imported model does not contain a {drop_class}!\n----"
        else:
            # figure which to choose
            if drop_comp == "auto":
                # pick the best component
                choosen = resolve_class(possible)[0]
            else:
                # user specified
                if not drop_comp in possible:
                    error = f"### ERROR: Imported model does not contain a {drop_comp}!\n----"
                else:
                    choosen = drop_comp
    
    reports = [gr.update(), gr.update()]
    names = [gr.update(), gr.update()]

    if not error:
        # delete the other conflicting components
        delete_class(loaded.model, drop_arch, drop_class)

        extract_component(model, choosen)

        replace_component(loaded.model, drop_arch, model, choosen)

        # update analysis
        filename = loaded.filename
        loaded = do_analysis(loaded.model)
        loaded.filename = filename

        # update reports and names
        result = do_report(precision)
        reports = [result[0], result[1]]
        names[0] = result[2]
        names[1] = get_name(loaded, drop_class)
    
    sources = [gr.update(), gr.update()]
    drops = [gr.update() for _ in range(3)]
    rows = [gr.update(), gr.update()]
    error = [gr.update(value=error), gr.update(visible=not not error)]

    updates = reports + sources + drops + rows + names + error
    return updates

def on_ui_tabs():
    get_lists()
    css = """
        .float-text { float: left; } .float-text-p { float: left; line-height: 2.5rem; } #mediumbutton { max-width: 32rem; } #smalldropdown { max-width: 2rem; } #smallbutton { max-width: 2rem; }
        #toolbutton { max-width: 8em; } #toolsettings > div > div { padding: 0; } #toolsettings { gap: 0.4em; } #toolsettings > div { border: none; background: none; gap: 0.5em; }
        #reportmd { padding: 1rem; } .dark #reportmd thead { color: #daddd8 } .gr-prose hr { margin-bottom: 0.5rem } #reportmd ul { margin-top: 0rem; margin-bottom: 0rem; } #reportmd li { margin-top: 0rem; margin-bottom: 0rem; }
        #errormd { min-height: 0rem; text-align: center; } #errormd h3 { color: #ba0000; }
    """
    with gr.Blocks(css=css, analytics_enabled=False, variant="compact") as checkpoint_toolkit:
        gr.HTML(value=f"<style>{css}</style>")
        with gr.Row() as load_row:
            source_dropdown = gr.Dropdown(label="Source", choices=source_list, value=source_list[0], interactive=True)
            load_button = gr.Button(value='Load', variant="primary")
            load_refresh_button = gr.Button(elem_id="smallbutton", value="Refresh")
        with gr.Row(visible=False) as save_row:
            save_name = gr.Textbox(label="Name", interactive=True)
            prec_dropdown = gr.Dropdown(elem_id="smalldropdown", label="Precision", choices=["FP16", "FP32"], value="FP16", interactive=True)
            save_button = gr.Button(value='Save', variant="primary")
            clear_button = gr.Button(elem_id="smallbutton", value="Clear")
            save_refresh_button = gr.Button(elem_id="smallbutton", value="Refresh")
        with gr.Row(visible=False) as error_row:
            error_md = gr.Markdown(elem_id="errormd", value="")
        with gr.Tab("Basic"):
            with gr.Column(variant="compact"):
                basic_report_md = gr.Markdown(elem_id="reportmd", value="")
        with gr.Tab("Advanced"):
            with gr.Column(variant="panel"):
                gr.HTML(value='<h1 class="gr-button-lg float-text">Component</h1><p class="float-text-p"><i>Select a component class or specific component.</i></p>')
                with gr.Row():
                    arch_dropdown = gr.Dropdown(label="Architecture", choices=[], interactive=True)
                    class_dropdown = gr.Dropdown(label="Class", choices=[], interactive=True)
                    comp_dropdown = gr.Dropdown(label="Component", choices=[], interactive=True)
                gr.HTML(value='<h1 class="gr-button-lg float-text">Action</h1><p class="float-text-p"><i>Replace or save the selected component.</i></p>')
                with gr.Row():
                    import_dropdown = gr.Dropdown(label="File", choices=file_list, value=file_list[0], interactive=True)
                    import_button = gr.Button(elem_id="smallbutton", value='Import')
                    export_name = gr.Textbox(label="Name", interactive=True)
                    export_button = gr.Button(elem_id="smallbutton", value='Export')
            with gr.Row(variant="compact"):
                adv_report_md = gr.Markdown(elem_id="reportmd", value="")

        reports = [basic_report_md, adv_report_md]
        sources = [source_dropdown, import_dropdown]
        drops = [arch_dropdown, class_dropdown, comp_dropdown]
        rows = [load_row, save_row]
        error = [error_md, error_row]
        names = [save_name, export_name]

        everything = reports + sources + drops + rows + names + error

        load_button.click(fn=do_load, inputs=[source_dropdown, prec_dropdown], outputs=everything)
        clear_button.click(fn=do_clear, inputs=[], outputs=everything)
        load_refresh_button.click(fn=do_refresh, inputs=[], outputs=[source_dropdown, import_dropdown])
        save_refresh_button.click(fn=do_refresh, inputs=[], outputs=[source_dropdown, import_dropdown])
        prec_dropdown.change(fn=do_report, inputs=[prec_dropdown], outputs=reports)
        arch_dropdown.change(fn=do_select, inputs=drops, outputs=drops + [export_name])
        class_dropdown.change(fn=do_select, inputs=drops, outputs=drops + [export_name])
        comp_dropdown.change(fn=do_select, inputs=drops, outputs=drops + [export_name])
        save_button.click(fn=do_save, inputs=[save_name, prec_dropdown], outputs=everything)

        export_button.click(fn=do_export, inputs=drops+[export_name], outputs=drops + [export_name] + error)
        import_button.click(fn=do_import, inputs=drops+[import_dropdown, prec_dropdown], outputs=everything)


    return (checkpoint_toolkit, "Toolkit", "checkpoint_toolkit"),

def on_ui_settings():
    section = ('model-toolkit', "Model Toolkit")
    shared.opts.add_option("model_toolkit_fix_clip", shared.OptionInfo(False, "Fix broken CLIP position IDs", section=section))

script_callbacks.on_ui_settings(on_ui_settings)

script_callbacks.on_ui_tabs(on_ui_tabs)

load_components(os.path.join(sys.path[0], "components"))