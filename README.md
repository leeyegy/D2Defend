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

## PREPARATION 
We provide five pre-trained models based on CIFAR10 [(Link)](https://drive.google.com/file/d/19334jRulgnpYSRQseK6BTHdlTFHTY8X6/view?usp=sharing, https://drive.google.com/file/d/1LQ8BUuSCUFSUz99KrKQ1C8fyzk17wYm9/view?usp=sharing, https://drive.google.com/file/d/1Nc7rijPeJzbPRUrF7zZ5cHt0mmEWmeJC/view?usp=sharing, https://drive.google.com/file/d/1kePpH-4x2OfH0Awuxeu_xQCq0PibZNxr/view?usp=sharing, https://drive.google.com/file/d/1y1dn8s86pNRKYIjvFqtuWRPB60JHDHK-/view?usp=sharing). You can download them and put them into ***./checkpoint/***(IF this path doesn't exist, then create it in advance.)


## WHOLE STEPS TO RE-PRODUCE OUR PAPER
### Step one. Generate adv file

```shell
bash demo_generate_adv_file.sh
```
You can change the attack setting(e.g., attack method and perturbation range ***epsilon***) in ***demo_genrate_adv_file.sh***.

### Step two. Generate imgs based on adv_file
To evaluate the adversarial noise level, D2Defend adopts a matlab program. To run this program, all adversarial examples need to be saved as image file firstly. 

```shell
bash demo_generate_image.sh
```
You can change the setting(e.g., attack method and perturbation range ***epsilon***) in ***demo_generate_image.sh***.

### Step three. Adversarial Noise estimation

```shell
matlab
run demo.m
```
By following the above steps, some \*.json will be created in the current dir, just move them into  ***./data/threshold_20/***(IF this path doesn't exist, then create them in advance.)

```shell
mv *.json ./data/threshold_20/
```

### Step Four. Generate adaptive sigma data file 

```shell
bash demo_generate_ada_sigma.sh
```

### Step Five. Run D2Defend

```shell
bash demo_adaptive_cifar10.sh
```

### Fixed Hyper-parameter Version
>Note: In this version, the key hyper-parameter ***sigma*** is fixed empirically, which may performs not very well against part of attack setting.
