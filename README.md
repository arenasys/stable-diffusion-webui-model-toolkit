# stable-diffusion-model-toolkit

A Multipurpose toolkit for managing, editing and creating models. 

Install by: `Extenstions tab > Install from URL > Paste in this pages URL > Install`

## Features
- Cleaning/pruning models.
- Converting to/from safetensors.
- Extracting/replacing model components.
- Identifying/debugging model architectures.

# Example
Many models being distributed are quite bloated, most of their size being redundant or useless data.

For example, Anything-v3.0 is 7.7gb and requires a separate 800mb VAE. These can be combined and cleaned into a **2.1gb standalone model**, with the correct VAE included.

Easy way, replace the VAE directly:
```
1. Select Anything-v3.0.ckpt from the source dropdown, press Load.
2. Change to the Advanced tab.
3. Select the class VAE-v1 and leave the component on auto
4. Select Anything-v3.0.vae.ckpt from the import dropdown, press Import.
5. Change the model name to something appropriate like Anything-v3.0.safetensors
6. Press Save.
```

Hard way, build the model from components:
```
1. Select Anything-v3.0.ckpt from the dropdown, press Load.
2. Change to the Advanced tab.
3. Select the class CLIP-v1, press Export.
4. Select the class UNET-v1, press Export.
5. Press Clear
6. Select NEW SDv1 from the source dropdown, press Load.
7. Select the class CLIP-v1, and select the Anything-v3 CLIP (just exported), press Import.
8. Select the class UNET-v1, and select the Anything-v3 UNET (just exported), press Import.
9. Select the class VAE-v1, and select the Anything-v3 VAE, press Import.
10. Change the model name to something appropriate like Anything-v3.0.safetensors
11. Press Save.
```

## Notes
Some things that may be useful to know when manipulating models.
### Precision
Comparison between the 2.1gb model and the orginal 8.5gb model.
**By default the webui will cast all loaded models to FP16**. Without `--no-half` the models will be exactly identical.
Shown is a comparison with `--no-half` enabled.
![](https://cdn.discordapp.com/attachments/973151736946622467/1060445743707603035/comparison.png)
But which is FP16 and which is FP32?

### EMA
EMA data is also often left in, which is only stored to enable training to stop and start as needed without losing momentum.

Interestingly, the EMA data is itself an independant and functional model, it can be extracted into its own ckpt and used. You can export the "UNET-v1-EMA" component to extract the EMA unet, then replace the regular UNET with it. For example the EMA UNET vs normal UNET in AnythingV3.
![](https://cdn.discordapp.com/attachments/973151736946622467/1060767681692827718/ema.png)

### CLIP
During merging a CLIP key called `embeddings.position_ids` is sometimes broken. This is an int64 tensor that has the values from 0 to 76, merging will convert these to float and introduce errors. For example in AnythingV3 the value `76` has become `75.9975`, which is cast back to int64 when loaded by the webui, resulting in `75`. The option `Fix broken CLIP position IDs` will fix this tensor, which changes the model output slightly (perhaps for the worse). Fixed vs Broken.
![](https://cdn.discordapp.com/attachments/973151736946622467/1060777823624765470/clip_fix.png)

### Merging
The UNET somehow takes to merging quite well, but thats not the case for the VAE. Any merge between different VAE's will result in something broken.
This is why the VAE in the AnythingV3 checkpoint produces horrible outputs, and why people have resorted to distributing and loading the VAE seperatly. 

But replacing the VAE is not a perfect solution. UNETs operate inside the latent space produced by the VAE. We dont know the effect merging has on the latent space the UNET is expecting, theres no reason it would interpolate with the weights. Luckily VAEs produce very similar latent spaces (by design), so in practice using one of the original VAEs will do a decent job.

The effect of merging on CLIP is currently unknown to me, but evidentily its not so devistating.
