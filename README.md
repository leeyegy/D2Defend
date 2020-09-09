# D2Defend
## REQUIREMENTS
> matlab

> PyTorch

> advertorch

> numpy

> cv2

> h5py

> json

> skimage

## QUICK START

## WHOLE STEPS TO RE-PRODUCE OUR PAPER
### Step one. Generate adv file

```shell
bash demo_generate_adv_file.sh
```
You can change the attack setting(e.g., attack method and perturbation range ***epsilon***) in ***demo_genrate_adv_file.sh***.

### Step two. Generate imgs based on adv_file
To evaluate the adversarial noise level, D2Defend adopts a matlab program. To run this program, all adversarial examples need to be saved as image file firstly. 

```shell
bash demo_
```



### Fixed Hyper-parameter Version
>Note: In this version, the key hyper-parameter ***sigma*** is fixed empirically, which may performs not very well against part of attack setting.
